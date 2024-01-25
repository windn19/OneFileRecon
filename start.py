from detect import load_model, run
try:
    from settings import yolo_weights
except Exception as e:
    print('Нет файла настроек с весами для yolo')
    yolo_weights = ''


def get_result(path):
    model = load_model(weights=yolo_weights)
    return run(model=model, source=path, save_crop=True, nosave=True)


