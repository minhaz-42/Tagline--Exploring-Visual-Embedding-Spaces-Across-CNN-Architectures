"""
forms.py
--------
Custom forms for user registration.
"""

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class RegisterForm(UserCreationForm):
    """Extended registration form with optional email."""

    email = forms.EmailField(required=False, widget=forms.EmailInput(attrs={
        "class": "form-input",
        "placeholder": "your@email.com",
    }))

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Apply consistent styling to all fields
        for field_name, field in self.fields.items():
            field.widget.attrs.setdefault("class", "form-input")
