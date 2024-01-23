ocr_file = ''

from detect import load_model, run


model = load_model(weights='')

print(run(model=model, source='test_image_part1/', save_crop=True, nosave=True))


