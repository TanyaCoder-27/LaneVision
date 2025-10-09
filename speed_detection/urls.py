from django.urls import path
from . import views
from django.contrib.auth.views import LogoutView

urlpatterns = [
    path('', views.home, name='home'),
    path('verify-email/', views.verify_email, name='verify_email'),
    path('verify-otp/', views.verify_otp, name='verify_otp'),
    path('signup/', views.signup, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    path('accounts/logout/', views.logout_view, name='accounts_logout'),
    path('contact/', views.contact_view, name='contact'),
    path('upload/', views.upload_video, name='upload_video'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('process/<int:video_id>/', views.process_video, name='process_video'),
    path('process-with-progress/<int:video_id>/', views.process_video_with_progress, name='process_video_with_progress'),
    path('progress/<int:video_id>/', views.get_processing_progress, name='get_processing_progress'),
    path('download-csv/<int:video_id>/', views.download_csv, name='download_csv'),
    path('video/<int:video_id>/', views.video_details, name='video_details'),
    path('delete/<int:video_id>/', views.delete_video, name='delete_video'),
]
