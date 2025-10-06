# Vehicle Speed Detection System

![Vehicle Speed Detection](https://imagevision.ai/wp-content/uploads/2024/01/Overspeeding-Detection-Section-01.jpg)

An advanced AI-powered system for detecting vehicle speeds and license plates in traffic videos with high accuracy. This web application uses computer vision and deep learning to analyze traffic footage, identify vehicles, track their movement, calculate speeds, and detect license plates.

## Features

- **Vehicle Detection & Tracking**: Identifies and tracks vehicles across video frames using YOLOv8
- **Speed Calculation**: Accurately measures vehicle speeds in km/h
- **License Plate Recognition**: Detects and reads license plates using OCR technology
- **Overspeed Detection**: Flags vehicles exceeding the speed limit (currently set to 80 km/h)
- **Video Processing**: Processes uploaded videos with visual indicators for detected vehicles
- **Data Export**: Generates CSV reports with detailed detection information
- **User Authentication**: Secure signup and login with email verification and OTP validation
- **Contact Form**: Built-in contact system for user support and inquiries
- **Dashboard**: User-friendly interface to manage and view processed videos
- **Responsive Design**: Works on desktop and mobile devices

## Technology Stack

- **Backend**: Django 5.0
- **Computer Vision**: OpenCV, YOLOv8 (Ultralytics)
- **OCR**: EasyOCR for license plate recognition
- **Data Processing**: NumPy, Pandas
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Authentication**: Django Authentication System with OTP verification

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd vehicle-speed-detection-New
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the YOLOv8 model (if not included):
   ```bash
   # The application will automatically download the model on first run
   # or you can manually place yolov8n.pt in the project root directory
   ```

5. Apply database migrations:
   ```bash
   python manage.py migrate
   ```

6. Create a superuser (admin):
   ```bash
   python manage.py createsuperuser
   ```

7. Run the development server:
   ```bash
   python manage.py runserver
   ```

8. Configure email settings for OTP verification:

   Option 1: Directly in settings.py:
   ```python
   # Email configuration
   EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
   EMAIL_HOST = 'smtp.your-email-provider.com'
   EMAIL_PORT = 587
   EMAIL_USE_TLS = True
   EMAIL_HOST_USER = 'your-email@example.com'
   EMAIL_HOST_PASSWORD = 'your-email-password'
   DEFAULT_FROM_EMAIL = 'your-email@example.com'
   ```

   Option 2: Using environment variables (recommended):
   
   Create a .env file in the project root:
   ```
   EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
   EMAIL_HOST=smtp.your-email-provider.com
   EMAIL_PORT=587
   EMAIL_USE_TLS=True
   EMAIL_HOST_USER=your-email@example.com
   EMAIL_HOST_PASSWORD=your-email-password
   DEFAULT_FROM_EMAIL=your-email@example.com
   ```
   
   Then update settings.py to use python-decouple:
   ```python
   from decouple import config
   
   EMAIL_BACKEND = config('EMAIL_BACKEND')
   EMAIL_HOST = config('EMAIL_HOST')
   EMAIL_PORT = config('EMAIL_PORT', cast=int)
   EMAIL_USE_TLS = config('EMAIL_USE_TLS', cast=bool)
   EMAIL_HOST_USER = config('EMAIL_HOST_USER')
   EMAIL_HOST_PASSWORD = config('EMAIL_HOST_PASSWORD')
   DEFAULT_FROM_EMAIL = config('DEFAULT_FROM_EMAIL')
   ```

9. Access the application at http://127.0.0.1:8000/

## Usage

### User Registration

1. Navigate to the signup page
2. Enter your email address for verification
3. Enter the OTP sent to your email
4. Complete the registration form

### Uploading and Processing Videos

1. Log in to your account
2. Click on "Upload Video" in the navigation or dashboard
3. Fill in the video title and upload your traffic video file
4. Wait for the processing to complete (progress will be displayed)
5. View the processed video with speed and license plate detections

### Viewing Results

1. Access your dashboard to see all uploaded videos
2. Click on a video to view detailed results
3. Download the CSV report for comprehensive detection data
4. Watch the processed video with visual indicators for detected vehicles

## System Architecture

### Components

- **Video Processing Engine**: Handles vehicle detection, tracking, and speed calculation
- **License Plate Recognition Module**: Extracts and reads license plate text
- **Web Interface**: Provides user interaction and result visualization
- **Database**: Stores user data, video metadata, and detection results

### Processing Pipeline

1. Video upload and validation
2. Frame-by-frame processing with YOLOv8 for vehicle detection
3. Vehicle tracking across frames
4. Speed calculation based on pixel displacement and calibration
5. License plate detection and OCR
6. Results storage and visualization

## Configuration

The system can be configured by modifying the following parameters in `video_processor.py`:

- `speed_limit`: Currently set to 80 km/h
- `pixels_per_meter`: Calibration parameter for speed calculation
- `vehicle_classes`: Classes of vehicles to detect [2, 3, 5, 7] (car, motorcycle, bus, truck)
- `speed_multiplier`: Adjustment factor for speed calculation (currently 1.2)

## Performance Optimization

- The system uses asynchronous video processing to prevent UI blocking
- Progress tracking is implemented for better user experience
- Vehicle tracking algorithm is optimized for accuracy and performance
- The application uses Django's built-in caching for improved response times

## Future Enhancements

- Real-time video processing from camera feeds
- Integration with traffic management systems
- Advanced analytics and reporting features
- Mobile application for on-the-go monitoring
- Multi-language support

## Security Considerations

- The default Django secret key should be changed in production
- Email credentials should be stored securely using environment variables
- Debug mode should be disabled in production
- HTTPS should be enabled for production deployments

## Troubleshooting

### Common Issues

1. **YOLOv8 Model Not Found**
   - Ensure the YOLOv8 model file (yolov8n.pt) is in the project root directory
   - Check internet connection if the model needs to be downloaded

2. **Email Verification Not Working**
   - Verify email settings in settings.py or .env file
   - Check if the email provider allows SMTP access
   - Try using a different email provider or enable "Less secure apps" if using Gmail

3. **Video Processing Errors**
   - Ensure the uploaded video is in a supported format (MP4 recommended)
   - Check if OpenCV is properly installed
   - Verify that the video contains traffic footage with vehicles

4. **Slow Processing Speed**
   - Processing time depends on video length and resolution
   - Consider using a machine with GPU support for faster processing
   - Try processing shorter video clips for testing

## License

[Specify your license here]

## Contributors

[List contributors here]

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for license plate recognition
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [Django](https://www.djangoproject.com/) for the web framework