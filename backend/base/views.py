from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def test(request):
    if request.method == "POST":
        return HttpResponse("Hello world!")
