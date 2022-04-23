from rest_framework.response import Response

from core.platform.filter import PlatformFilter
from core.util.serializable import from_json
from rest.rest_view import RestView


class AnalysisView(RestView):

    def post(self, request, format=None):
        filters = [from_json(f, PlatformFilter, f['platform']) for f in request.data]

        return Response(self.platform_facade.analyze(filters))
