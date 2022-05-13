import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

from main import video_processing

def on_created(event):
    print(f"{event.src_path} has been created!")
    temp_str = str(event.src_path)
    video_file = temp_str.partition("/")[2]
    video_name = video_file.split(".")[0]
    print("Video name is: " + video_name)
    video_processing(video_name)

if __name__ == "__main__":
    patterns = ["*"]
    ignore_patterns = None
    ignore_directories = False
    case_sensitive = True
    my_event_handler = PatternMatchingEventHandler(patterns, ignore_patterns, ignore_directories, case_sensitive)

    my_event_handler.on_created = on_created

    path = "unprocessedFiles/"
    go_recursively = True
    my_observer = Observer()
    my_observer.schedule(my_event_handler, path, recursive=go_recursively)

    my_observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        my_observer.stop()
        my_observer.join()
