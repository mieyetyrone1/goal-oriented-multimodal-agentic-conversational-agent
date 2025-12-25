class AudioController:
    def __init__(self):
        self.state = "IDLE"

    def begin_listening(self):
        if self.state != "IDLE":
            raise RuntimeError("Cannot listen while speaking")
        self.state = "LISTENING"
        print("Mic ON")

    def end_listening(self):
        if self.state != "LISTENING":
            return
        self.state = "IDLE"
        print("Mic OFF")

    def begin_speaking(self):
        if self.state != "IDLE":
            raise RuntimeError("Cannot speak while listening")
        self.state = "SPEAKING"
        print("Speaker ON")

    def end_speaking(self):
        if self.state != "SPEAKING":
            return
        self.state = "IDLE"
        print("Speaker OFF")

    def can_listen(self):
        return self.state == "LISTENING"
