# Conveyor homework

В input_path нужно передать путь до файла с сгенерированными тестовыми данными.
В output_path нужно передать путь до двух папок. Одна должна называться gt, а другая preds. В одну папку будут складываться оригинальные изображения, а во вторую предсказанные.

Посчитать скор можно будет с помощью команды: python evaluate.py ./output_path/gt/ ./output_path/preds/
