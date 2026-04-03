from src.version import maki
from src.version import shin
from src.version import snoopy


def draw_maki():
    maki.process_maki_img()


def draw_shin(clean_manual=False):
    if clean_manual:
        shin.clean_canny_edges_manual()
    shin.process_shin_gif()


def draw_snoopy(clean_manual=False):
    if clean_manual:
        snoopy.clean_canny_edges_manual()


if __name__ == "__main__":
    # draw_shin(clean_manual=False)
    # draw_maki()
    draw_snoopy(clean_manual=True)
