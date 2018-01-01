import json


class JSONLogger(object):

    def __init__(self, filedir, filename, config):
        self.filepath = os.path.join(filedir, filename) + ".json"
        self.log = {"config" config:, "events": []}

    def add_event(self, event):
        self.log["events"].append(event)

    def write_logs(self):
        with open(self.filepath, "w+") as fp:
            json.dump(self.log, fp)
