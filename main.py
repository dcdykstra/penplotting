from src.version import maki
from src.version import shin
from src.version import snoopy


def draw_maki():
    maki.process_maki_img()


def draw_shin(clean_manual=False):
    if clean_manual:
        shin.clean_shin_edges_manual()
    shin.process_shin_gif()


def draw_snoopy(clean_manual=False):
    if clean_manual:
        snoopy.clean_snoopy_edges_manual()
    snoopy.process_snoopy_gif()


if __name__ == "__main__":
    print("Processing shin...")
    draw_shin(clean_manual=False)
    print("Processing maki...")
    draw_maki()
    print("Processing snoopy.,,")
    draw_snoopy(clean_manual=False)
