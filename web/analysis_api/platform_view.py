from rest_framework.response import Response

from core.platform.supported_platforms import SupportedPlatform
from rest.rest_view import RestView


class PlatformView(RestView):

    def get(self, request, format=None):
        return Response(SupportedPlatform.to_dict())
