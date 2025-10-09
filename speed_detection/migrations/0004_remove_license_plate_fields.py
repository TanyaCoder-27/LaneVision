# Generated migration to remove license plate detection functionality

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('speed_detection', '0003_vehicledetection_license_plate_confidence'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='vehicledetection',
            name='license_plate',
        ),
        migrations.RemoveField(
            model_name='vehicledetection',
            name='license_plate_confidence',
        ),
    ]