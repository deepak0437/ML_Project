"""Diabetes URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
#from django.conf.urls import url
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.index),
    path('predict/', views.predict),
    path('predict/result', views.result),
    path('marks/', views.marks),
    path('marks/result_marks', views.result_marks),
    path('wine/', views.wine),
    path('wine/result_wine', views.result_wine),
    path('house/', views.house),
    path('house/result_house', views.result_house),
    path('insurance/', views.insurance),
    path('insurance/result_insurance', views.result_insurance),
    path('cement/', views.cement),
    path('cement/result_cement', views.result_cement),
    path('require/', views.require)
    
]
