from rest_framework.response import Response

from rest.rest_view import RestView


class FiltersView(RestView):

    def get(self, request, format=None):
        return Response(self._platform_facade.get_filter(request.query_params['platform']).get_filters())
