# Generated by Django 4.1.2 on 2022-10-13 09:22

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("about_us", "0001_initial"),
    ]

    operations = [
        migrations.RenameModel(
            old_name="Movies",
            new_name="Movie",
        ),
    ]