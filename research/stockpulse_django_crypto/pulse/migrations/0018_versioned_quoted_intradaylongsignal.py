from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("pulse", "0017_intradaylongsignal"),
    ]

    operations = [
        migrations.AddField(
            model_name="intradaylongsignal",
            name="strategy_version",
            field=models.CharField(default="mtf_breakout_v1_unquoted", max_length=64),
        ),
        migrations.AddField(
            model_name="intradaylongsignal",
            name="entry_quote_observed_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="intradaylongsignal",
            name="entry_quote_spread_pct",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.RemoveConstraint(
            model_name="intradaylongsignal",
            name="unique_intraday_long_signal_per_closed_trigger",
        ),
        migrations.AddConstraint(
            model_name="intradaylongsignal",
            constraint=models.UniqueConstraint(
                fields=("contract_symbol", "profile", "strategy_version", "signal_opened_at"),
                name="unique_intraday_long_signal_per_closed_trigger_and_version",
            ),
        ),
    ]
