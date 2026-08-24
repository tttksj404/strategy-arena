from __future__ import annotations

from dataclasses import dataclass

from django.db import models

from .models import TimeStampedModel


SOURCE_HISTORICAL = "historical_exploratory"
SOURCE_FORWARD = "forward"
SOURCE_CHOICES = ((SOURCE_HISTORICAL, "Historical exploratory"), (SOURCE_FORWARD, "Forward"))
DIRECTION_CHOICES = (("UP", "Up"), ("DOWN", "Down"))
OUTCOME_CHOICES = (("success", "Success"), ("failure", "Failure"))
PHASE_CHOICES = (("validation", "Validation"), ("holdout", "Holdout"))


@dataclass(frozen=True, slots=True)
class ImmutableTrendContinuationError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


class ImmutableTrendContinuationQuerySet(models.QuerySet):
    def update(self, **kwargs):  # noqa: ANN003
        raise ImmutableTrendContinuationError("trend-continuation rows cannot be bulk updated")

    def delete(self):
        raise ImmutableTrendContinuationError("trend-continuation rows cannot be bulk deleted")

    def bulk_update(self, objs, fields, batch_size=None):  # noqa: ANN001
        raise ImmutableTrendContinuationError("trend-continuation rows cannot be bulk updated")

    def bulk_create(self, objs, **kwargs):  # noqa: ANN001, ANN003
        rows = tuple(objs)
        for row in rows:
            row.full_clean()
        return super().bulk_create(rows, **kwargs)


ImmutableTrendContinuationManager = models.Manager.from_queryset(ImmutableTrendContinuationQuerySet)


class ImmutableModelMixin(TimeStampedModel):
    objects = ImmutableTrendContinuationManager()

    class Meta:
        abstract = True

    def save(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if self.pk is not None and not self._state.adding:
            raise ImmutableTrendContinuationError("trend-continuation rows are immutable")
        self.full_clean()
        return super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise ImmutableTrendContinuationError("trend-continuation rows are append-only")
