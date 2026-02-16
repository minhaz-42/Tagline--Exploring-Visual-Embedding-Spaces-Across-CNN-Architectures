"""
models.py
---------
Django models for the retrieval app.

This project uses a file-based pipeline (numpy arrays, JSON) rather than a
relational database for embeddings.  The Django model below is kept minimal
– it simply records uploaded query images so they can be cleaned up later.
"""

from django.conf import settings
from django.db import models


class QueryImage(models.Model):
    """Stores metadata about user-uploaded query images."""

    image = models.ImageField(upload_to="queries/%Y/%m/%d/")
    model_key = models.CharField(max_length=20)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self) -> str:
        return f"Query({self.model_key}) @ {self.uploaded_at:%Y-%m-%d %H:%M}"


class SearchHistory(models.Model):
    """Records each search query for the history/dashboard pages."""

    MODEL_CHOICES = [
        ("resnet", "ResNet-101"),
        ("zfnet", "ZFNet"),
        ("googlenet", "GoogLeNet"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="search_history",
        null=True,
        blank=True,
    )
    model_key = models.CharField(max_length=20, choices=MODEL_CHOICES)
    query_image_filename = models.CharField(max_length=255, blank=True)
    top_result_class = models.CharField(max_length=100, blank=True)
    top_similarity = models.FloatField(null=True, blank=True)
    result_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name_plural = "Search histories"

    def __str__(self) -> str:
        user_str = self.user.username if self.user else "anonymous"
        return f"Search({self.model_key}) by {user_str} @ {self.created_at:%Y-%m-%d %H:%M}"

    @property
    def model_display(self) -> str:
        return dict(self.MODEL_CHOICES).get(self.model_key, self.model_key)

    @property
    def query_image_url(self) -> str:
        if self.query_image_filename:
            return f"uploads/{self.query_image_filename}"
        return ""
