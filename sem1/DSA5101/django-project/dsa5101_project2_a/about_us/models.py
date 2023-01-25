from django.db import models


class Movie(models.Model):
    name = models.CharField(max_length=255)
    duration = models.IntegerField()
    grade = models.CharField(max_length=255)
    image_url = models.CharField(max_length=2083)
