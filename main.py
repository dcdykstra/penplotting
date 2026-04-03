from src.version import maki
from src.version import shin


def draw_maki():
    maki.process_maki_img()


def draw_shin(clean_manual=False):
    if clean_manual:
        shin.clean_canny_edges_manual()
    shin.process_shin_gif()


if __name__ == "__main__":
    draw_shin(clean_manual=False)
    draw_maki()
