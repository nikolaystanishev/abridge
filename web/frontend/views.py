from django.shortcuts import render

from core.util.sh import register_postactions


def index(request, *args, **kwargs):
    return render(request, 'frontend/src/index.html')
