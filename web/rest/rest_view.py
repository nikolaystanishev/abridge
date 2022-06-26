from abc import ABC

from rest_framework.views import APIView

from core.platform.platform_facade import PlatformFacade


class RestView(APIView, ABC):

    def __init__(self):
        self._platform_facade: PlatformFacade = PlatformFacade()
