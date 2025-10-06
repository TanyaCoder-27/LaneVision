from django.contrib import admin
from .models import UploadedVideo, VehicleDetection, Contact

@admin.register(UploadedVideo)
class UploadedVideoAdmin(admin.ModelAdmin):
    list_display = ['title', 'user', 'uploaded_at', 'is_processed']
    list_filter = ['is_processed', 'uploaded_at']
    search_fields = ['title', 'user__username']
    readonly_fields = ['uploaded_at']
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')

@admin.register(VehicleDetection)
class VehicleDetectionAdmin(admin.ModelAdmin):
    list_display = ['video', 'vehicle_id', 'speed', 'is_overspeed', 'frame_number']
    list_filter = ['is_overspeed', 'video', 'timestamp']
    search_fields = ['video__title']
    readonly_fields = ['timestamp']
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('video', 'video__user')

@admin.register(Contact)
class ContactAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'subject', 'created_at']
    list_filter = ['created_at']
    search_fields = ['name', 'email', 'subject']
    readonly_fields = ['created_at']
