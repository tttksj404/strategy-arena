from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("pulse", "0016_communitypost_source_fingerprint_constraint"),
    ]

    operations = [
        migrations.CreateModel(
            name="IntradayLongSignal",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("contract_symbol", models.CharField(max_length=32)),
                ("underlying_symbol", models.CharField(max_length=24)),
                ("profile", models.CharField(max_length=48)),
                ("signal_opened_at", models.DateTimeField()),
                ("starts_at", models.DateTimeField()),
                ("expires_at", models.DateTimeField()),
                ("entry_price", models.DecimalField(decimal_places=8, max_digits=24)),
                ("target_price", models.DecimalField(decimal_places=8, max_digits=24)),
                ("stop_price", models.DecimalField(decimal_places=8, max_digits=24)),
                ("taker_fee_rate", models.FloatField()),
                ("estimated_margin_loss_at_stop_pct", models.FloatField()),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("pending", "Pending"),
                            ("target", "Target"),
                            ("stop", "Stop"),
                            ("unresolved", "Unresolved"),
                            ("ambiguous", "Ambiguous"),
                        ],
                        default="pending",
                        max_length=16,
                    ),
                ),
                ("terminal_price", models.DecimalField(blank=True, decimal_places=8, max_digits=24, null=True)),
                ("post_cost_margin_return_pct", models.FloatField(blank=True, null=True)),
                ("settled_at", models.DateTimeField(blank=True, null=True)),
            ],
            options={"ordering": ("-signal_opened_at", "contract_symbol")},
        ),
        migrations.AddConstraint(
            model_name="intradaylongsignal",
            constraint=models.UniqueConstraint(
                fields=("contract_symbol", "profile", "signal_opened_at"),
                name="unique_intraday_long_signal_per_closed_trigger",
            ),
        ),
    ]
