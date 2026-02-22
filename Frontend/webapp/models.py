from django.db import models
import os

# Create your models here.


class eye(models.Model):
    images = models.ImageField(upload_to='webapp/static/image')

    def filename(self):
        return os.path.basename(self.images.name)
