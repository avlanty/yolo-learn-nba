from utils import read_vid, save_vid
from trackers import Tracker

def main():
    # read the vid
    vid_frames = read_vid('videos/murraygw.mp4')

    # init tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(vid_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # draw output
    output_vid_frames = tracker.draw_annotations(vid_frames, tracks)

    # save the vid
    save_vid(output_vid_frames, 'out_videos/out_vid.avi')

if __name__ == '__main__':
    main()