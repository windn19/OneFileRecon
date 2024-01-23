from datetime import datetime

from peewee import SqliteDatabase, Model, DateTimeField, CharField, TextField


db = SqliteDatabase('../base.db')


class Numbers(Model):
    datetime = DateTimeField(primary_key=True, default=datetime.now())
    num = CharField(max_length=9)
    crop = CharField(max_length=30)
    image = CharField(max_length=30)
    res = TextField()
    cam_name = CharField(max_length=10)

    class Meta:
        database = db
        table_name = 'base'


db.connect()

