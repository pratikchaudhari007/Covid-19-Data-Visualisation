from django.contrib import admin
from django.urls import path,include 
from . import views
urlpatterns = [
  path('', views.index, name='index'),
  path('index1',views.index1),
  path('india',views.india),
  path('world',views.world),
  path('us',views.us),
  path('russia',views.russia),
  path('ind',views.ind),
  path('brazil',views.brazil),
  path('sa',views.sa),
  path('china',views.china),
  path('spain',views.spain),
  path('italy',views.italy),
  path('germany',views.germany),
  path('france',views.france),
  
]
