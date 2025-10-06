from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from django.db import models
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
from django.core.mail import send_mail
from django.utils import timezone
from .forms import SignUpForm, VideoUploadForm, ContactForm, EmailVerificationForm, OTPVerificationForm
from .models import UploadedVideo, VehicleDetection, Contact, OTPVerification
from .video_processor import VehicleSpeedDetector
import os
import pandas as pd
import csv
import json
import threading
import time
import random

# Global variable to store processing progress
processing_progress = {}

def home(request):
    return render(request, 'home.html')

def logout_view(request):
    # Handle both GET and POST requests for logout
    logout(request)
    messages.success(request, 'You have been successfully logged out.')
    return redirect('home')

def send_otp_email(email, otp):
    subject = 'Your OTP for Vehicle Speed Detection System Registration'
    message = f'Your OTP for registration is: {otp}\n\nThis OTP is valid for 10 minutes.'
    from_email = settings.DEFAULT_FROM_EMAIL
    recipient_list = [email]
    send_mail(subject, message, from_email, recipient_list)


def verify_email(request):
    if request.method == 'POST':
        form = EmailVerificationForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            
            # Delete any existing OTPs for this email
            OTPVerification.objects.filter(email=email).delete()
            
            # Create new OTP
            otp_code = ''.join([str(random.randint(0, 9)) for _ in range(6)])
            expires_at = timezone.now() + timezone.timedelta(minutes=10)
            otp_obj = OTPVerification.objects.create(
                email=email,
                otp=otp_code,
                expires_at=expires_at
            )
            
            # Send OTP via email
            send_otp_email(email, otp_code)
            
            # Store email in session for next step
            request.session['verification_email'] = email
            
            messages.success(request, 'OTP has been sent to your email. Please verify to continue.')
            return redirect('verify_otp')
    else:
        form = EmailVerificationForm()
    
    return render(request, 'registration/verify_email.html', {'form': form})


def verify_otp(request):
    if 'verification_email' not in request.session:
        messages.error(request, 'Please provide your email first.')
        return redirect('verify_email')
    
    email = request.session['verification_email']
    
    if request.method == 'POST':
        form = OTPVerificationForm(request.POST)
        if form.is_valid():
            otp = form.cleaned_data['otp']
            
            try:
                otp_obj = OTPVerification.objects.get(email=email, otp=otp, is_verified=False)
                
                # Check if OTP is expired
                if timezone.now() > otp_obj.expires_at:
                    messages.error(request, 'OTP has expired. Please request a new one.')
                    return redirect('verify_email')
                
                # Mark OTP as verified
                otp_obj.is_verified = True
                otp_obj.save()
                
                # Store verification status in session
                request.session['email_verified'] = True
                
                messages.success(request, 'Email verified successfully! Please complete your registration.')
                return redirect('signup')
            
            except OTPVerification.DoesNotExist:
                messages.error(request, 'Invalid OTP. Please try again.')
    
    form = OTPVerificationForm()
    return render(request, 'registration/verify_otp.html', {'form': form, 'email': email})


def signup(request):
    # Check if email is verified
    if 'email_verified' not in request.session or not request.session['email_verified']:
        messages.error(request, 'Please verify your email first.')
        return redirect('verify_email')
    
    email = request.session.get('verification_email')
    
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            # Check if the email in the form matches the verified email
            if form.cleaned_data['email'] != email:
                messages.error(request, 'Please use the same email that was verified.')
                return render(request, 'registration/signup.html', {'form': form})
            
            user = form.save()
            
            # Clear verification session data
            if 'verification_email' in request.session:
                del request.session['verification_email']
            if 'email_verified' in request.session:
                del request.session['email_verified']
            
            login(request, user)
            messages.success(request, 'Account created successfully!')
            return redirect('home')
    else:
        form = SignUpForm(initial={'email': email})
    
    return render(request, 'registration/signup.html', {'form': form})

def contact_view(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Thank you for your message! We will get back to you soon.')
            return redirect('contact')
    else:
        form = ContactForm()
    return render(request, 'contact.html', {'form': form})

@login_required
def upload_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save(commit=False)
            video.user = request.user
            video.save()
            messages.success(request, 'Video uploaded successfully! Processing will begin shortly.')
            return redirect('dashboard')
    else:
        form = VideoUploadForm()
    return render(request, 'upload_video.html', {'form': form})

@login_required
def dashboard(request):
    videos = UploadedVideo.objects.filter(user=request.user).order_by('-uploaded_at')
    paginator = Paginator(videos, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    return render(request, 'dashboard.html', {'page_obj': page_obj})

def update_progress(video_id, progress, message):
    """Update processing progress"""
    processing_progress[video_id] = {
        'progress': progress,
        'message': message,
        'timestamp': time.time()
    }

@login_required
def get_processing_progress(request, video_id):
    """Get current processing progress"""
    progress_info = processing_progress.get(video_id, {
        'progress': 0,
        'message': 'Starting...',
        'timestamp': time.time()
    })
    return JsonResponse(progress_info)

@login_required
def process_video_with_progress(request, video_id):
    """Process video with real-time progress updates"""
    video = get_object_or_404(UploadedVideo, id=video_id, user=request.user)
    
    if video.is_processed:
        return JsonResponse({'error': 'Video is already processed!'}, status=400)
    
    def progress_callback(progress, message):
        update_progress(video_id, progress, message)
    
    def process_video_thread():
        try:
            # Initialize video processor with progress callback
            processor = VehicleSpeedDetector(progress_callback=progress_callback)
            
            # Define file paths
            input_path = video.video_file.path
            output_filename = f"processed_{video.id}_{video.video_file.name}"
            csv_filename = f"detections_{video.id}.csv"
            
            output_path = os.path.join(settings.MEDIA_ROOT, 'processed_videos', output_filename)
            csv_path = os.path.join(settings.MEDIA_ROOT, 'csv_files', csv_filename)
            
            # Ensure directories exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            
            # Update progress
            update_progress(video_id, 5, "Initializing AI models...")
            
            # Process video
            detection_count = processor.process_video(input_path, output_path, csv_path)
            
            # Update progress
            update_progress(video_id, 95, "Saving results...")
            
            # Update video object with correct path
            video.processed_video = f"processed_videos/{output_filename}"
            video.csv_file = f"csv_files/{csv_filename}"
            video.is_processed = True
            video.save()
            
            # Save detection results to database
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    VehicleDetection.objects.create(
                        video=video,
                        frame_number=row['frame_number'],
                        vehicle_id=row['vehicle_id'],
                        speed=row['speed'],
                        is_overspeed=row['is_overspeed'],
                        x1=row['x1'], y1=row['y1'], x2=row['x2'], y2=row['y2']
                    )
            
            # Final progress update
            update_progress(video_id, 100, f"Completed! Found {detection_count} vehicle detections.")
        
        except Exception as e:
            update_progress(video_id, -1, f"Error: {str(e)}")
    
    # Start processing in background thread
    thread = threading.Thread(target=process_video_thread)
    thread.daemon = True
    thread.start()
    
    return JsonResponse({'status': 'started', 'message': 'Video processing started'})

@login_required
def process_video(request, video_id):
    """Legacy process video function (without progress)"""
    video = get_object_or_404(UploadedVideo, id=video_id, user=request.user)
    
    if video.is_processed:
        messages.info(request, 'Video is already processed!')
        return redirect('dashboard')
    
    try:
        # Initialize video processor
        processor = VehicleSpeedDetector()
        
        # Define file paths
        input_path = video.video_file.path
        output_filename = f"processed_{video.id}_{video.video_file.name}"
        csv_filename = f"detections_{video.id}.csv"
        
        output_path = os.path.join(settings.MEDIA_ROOT, 'processed_videos', output_filename)
        csv_path = os.path.join(settings.MEDIA_ROOT, 'csv_files', csv_filename)
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Process video
        detection_count = processor.process_video(input_path, output_path, csv_path)
        
        # Update video object
        video.processed_video = f"processed_videos/{output_filename}"
        video.csv_file = f"csv_files/{csv_filename}"
        video.is_processed = True
        video.save()
        
        # Save detection results to database
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                VehicleDetection.objects.create(
                    video=video,
                    frame_number=int(row['frame_number']),
                    vehicle_id=int(row['vehicle_id']),
                    speed=float(row['speed']),
                    is_overspeed=row['is_overspeed'] == 'True',
                    x1=int(row['x1']), y1=int(row['y1']), x2=int(row['x2']), y2=int(row['y2'])
                )
        
        messages.success(request, f'Video processed successfully! {detection_count} vehicle detections found.')
    
    except Exception as e:
        messages.error(request, f'Error processing video: {str(e)}')
    
    return redirect('dashboard')

@login_required
def download_csv(request, video_id):
    video = get_object_or_404(UploadedVideo, id=video_id, user=request.user)
    
    if not video.csv_file:
        messages.error(request, 'CSV file not found!')
        return redirect('dashboard')
    
    try:
        with open(video.csv_file.path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = f'inline; filename={os.path.basename(video.csv_file.name)}'
            return response
    except FileNotFoundError:
        messages.error(request, 'CSV file not found!')
        return redirect('dashboard')

@login_required
def video_details(request, video_id):
    video = get_object_or_404(UploadedVideo, id=video_id, user=request.user)
    
    # Get all detections
    all_detections = VehicleDetection.objects.filter(video=video).order_by('frame_number')
    
    # Get unique vehicle IDs
    unique_vehicle_ids = all_detections.values_list('vehicle_id', flat=True).distinct()
    
    # For each unique vehicle ID, get only the detection with the highest speed
    unique_detections = []
    for vehicle_id in unique_vehicle_ids:
        # Get the detection with the highest speed for this vehicle ID
        best_detection = all_detections.filter(vehicle_id=vehicle_id).order_by('-speed').first()
        if best_detection:
            unique_detections.append(best_detection)
    
    # Statistics - use all detections for statistics
    total_vehicles = len(unique_vehicle_ids)  # Count unique vehicles
    overspeed_vehicles = all_detections.filter(is_overspeed=True).values('vehicle_id').distinct().count()
    
    # Calculate average speed
    avg_speed = all_detections.aggregate(avg_speed=models.Avg('speed'))['avg_speed'] or 0
    
    context = {
        'video': video,
        'detections': unique_detections,  # Use the filtered list with unique vehicle IDs
        'total_vehicles': total_vehicles,
        'overspeed_vehicles': overspeed_vehicles,
        'avg_speed': round(avg_speed, 2),
        'MEDIA_URL': settings.MEDIA_URL,
    }
    
    return render(request, 'video_details.html', context)

@login_required
def delete_video(request, video_id):
    video = get_object_or_404(UploadedVideo, id=video_id, user=request.user)
    
    # Delete associated files
    if video.video_file and os.path.exists(video.video_file.path):
        os.remove(video.video_file.path)
    if video.processed_video and os.path.exists(video.processed_video.path):
        os.remove(video.processed_video.path)
    if video.csv_file and os.path.exists(video.csv_file.path):
        os.remove(video.csv_file.path)
    
    video.delete()
    messages.success(request, 'Video deleted successfully!')
    return redirect('dashboard')



