def get_image_file_list(sample_set: dict) -> list:
    image_file_list = list()
    for category in sample_set.keys():
        for image_file in sample_set[category]['image_file']:
            image_file_list.append(image_file)
    return image_file_list
