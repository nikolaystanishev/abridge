from django.urls import path

from .analysis_view import AnalysisView
from .filter_view import FiltersView
from .platform_view import PlatformView

urlpatterns = [
    path('platform', PlatformView.as_view()),
    path('filter', FiltersView.as_view()),
    path('analysis', AnalysisView.as_view()),
]
