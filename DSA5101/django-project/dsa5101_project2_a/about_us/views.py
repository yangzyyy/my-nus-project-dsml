from django.shortcuts import render
from django.http import HttpResponse
from .models import Movie


def index(request):
    return render(request, 'index.html')


def movies(request):
    movies = Movie.objects.all()
    return render(request, 'movies.html', {'movies': movies})
