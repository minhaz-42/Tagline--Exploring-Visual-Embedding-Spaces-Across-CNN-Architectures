"""
URL configuration for the retrieval app.
"""

from django.urls import path
from . import views

urlpatterns = [
    # Public pages
    path("", views.landing, name="landing"),
    path("search/", views.search_page, name="search_page"),
    path("search/go/", views.search, name="search"),
    path("metrics/", views.metrics, name="metrics"),

    # Auth
    path("login/", views.user_login, name="user_login"),
    path("register/", views.user_register, name="user_register"),
    path("logout/", views.user_logout, name="user_logout"),

    # Authenticated pages
    path("dashboard/", views.dashboard, name="dashboard"),
    path("history/", views.history, name="history"),
]
