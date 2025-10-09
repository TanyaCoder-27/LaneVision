# LaneVision Deployment Guide

## 🚀 Free Deployment Options

### Option 1: Railway (Recommended)
1. **Sign up** at [railway.app](https://railway.app)
2. **Connect GitHub** and select your LaneVision repository
3. **Deploy from branch**: `yolov8m-realistic-fallback`
4. **Set environment variables**:
   - `DEBUG=False`
   - `SECRET_KEY=your-secret-key-here`
5. **Deploy!** Railway will automatically build and deploy your app

### Option 2: Render
1. **Sign up** at [render.com](https://render.com)
2. **Create new Web Service** from GitHub
3. **Select repository**: LaneVision
4. **Build settings**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn vehicle_detection.wsgi:application`
5. **Environment variables**:
   - `DEBUG=False`
   - `SECRET_KEY=your-secret-key-here`
6. **Deploy!**

### Option 3: PythonAnywhere
1. **Sign up** at [pythonanywhere.com](https://pythonanywhere.com)
2. **Upload your code** via Git or file upload
3. **Create Web App** with Django
4. **Configure WSGI file** to point to your project
5. **Install dependencies** in Bash console
6. **Reload** your web app

## 📋 Pre-deployment Checklist

- ✅ **Procfile** created for deployment
- ✅ **requirements.txt** updated with gunicorn and whitenoise
- ✅ **Django settings** configured for production
- ✅ **Static files** configured with WhiteNoise
- ✅ **Media files** properly configured
- ✅ **Environment variables** set for production

## 🔧 Important Notes

### File Size Considerations
- **YOLO models** (yolov8m.pt - 50MB) are large
- Consider using **Git LFS** for large files
- Or host models on **cloud storage** (AWS S3, etc.)

### Performance Optimization
- **CPU-only processing** will be slower on cloud
- Consider **GPU instances** for better performance
- **Frame skipping** already implemented for speed

### Security
- **SECRET_KEY** should be changed for production
- **DEBUG=False** in production
- **Email credentials** should use environment variables

## 🌐 Custom Domain (Optional)
- Most platforms support custom domains
- **Railway**: Free custom domain support
- **Render**: Custom domains available on paid plans

## 📊 Monitoring
- Monitor **CPU usage** and **memory consumption**
- Set up **error tracking** (Sentry, etc.)
- Monitor **video processing times**

## 🆘 Troubleshooting
- Check **build logs** for dependency issues
- Verify **environment variables** are set
- Ensure **static files** are collected properly
- Check **media file permissions**

---

**Ready to deploy?** Choose your platform and follow the steps above!
