import json
import os


class JSONLogger(object):

    def __init__(self, filedir, filename, config):
        self.filepath = os.path.join(filedir, filename) + ".json"
        self.log = {"config" config:, "events": []}
        self.new_file = os.path.isfile(self.filepath)

    def add_event(self, event):
        self.log["events"].append(event)

    def save_log(self):
        with open(self.filepath, "a+") as fp:
            if self.new_file:
                json.dump([self.log], fp)
            else:
                all_logs = json.load(fp)
                all_logs.append(self.log)
                json.dump(all_logs, fp)
