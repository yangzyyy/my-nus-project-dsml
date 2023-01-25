from django.contrib import admin
from .models import Movie


class MovieAdmin(admin.ModelAdmin):
    list_display = ('name', 'duration', 'grade')


admin.site.register(Movie, MovieAdmin)