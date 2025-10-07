from django.db import models
from django.contrib.auth.models import User
import os
import random
from datetime import datetime, timedelta

class UploadedVideo(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    video_file = models.FileField(upload_to='videos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_processed = models.BooleanField(default=False)
    processed_video = models.FileField(upload_to='processed_videos/', blank=True, null=True)
    csv_file = models.FileField(upload_to='csv_files/', blank=True, null=True)

    def __str__(self):
        return f"{self.title} - {self.user.username}"

class VehicleDetection(models.Model):
    video = models.ForeignKey(UploadedVideo, on_delete=models.CASCADE)
    frame_number = models.IntegerField()
    vehicle_id = models.IntegerField()
    speed = models.FloatField()
    is_overspeed = models.BooleanField(default=False)
    x1 = models.IntegerField()
    y1 = models.IntegerField()
    x2 = models.IntegerField()
    y2 = models.IntegerField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Vehicle {self.vehicle_id} - Speed: {self.speed} km/h"

class Contact(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    subject = models.CharField(max_length=200)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} - {self.subject}"


class OTPVerification(models.Model):
    email = models.EmailField()
    otp = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    is_verified = models.BooleanField(default=False)
    
    def __str__(self):
        return f"OTP for {self.email}"
    
    def save(self, *args, **kwargs):
        if not self.otp:
            self.otp = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        if not self.expires_at:
            self.expires_at = datetime.now() + timedelta(minutes=10)
        super().save(*args, **kwargs)
    
    @property
    def is_expired(self):
        return datetime.now() > self.expires_at
