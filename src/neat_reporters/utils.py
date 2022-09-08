import pygame


def aspect_scale(img, bx, by):
    """ Scales 'img' to fit into box bx/by.
     This method will retain the original image's aspect ratio """
    ix, iy = img.get_size()
    if ix > iy:
        # fit to width
        scale_factor = bx / ix
        sy = scale_factor * iy
        if sy > by:
            scale_factor = by / iy
            sx = scale_factor * ix
            sy = by
        else:
            sx = bx
    else:
        # fit to height
        scale_factor = by / iy
        sx = scale_factor * ix
        if sx > bx:
            scale_factor = bx / ix
            sx = bx
            sy = scale_factor * iy
        else:
            sy = by
    print(img.get_size())

    image = pygame.transform.scale(img, (int(sx), int(sy)))
    # blit the scaled image on a surface
    background = pygame.Surface((int(bx), int(by)))
    background.blit(image, (int((bx - sx) / 2), int((by - sy) / 2)))
    print(background.get_size())
    return background
