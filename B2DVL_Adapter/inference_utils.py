from datetime import datetime

class Context:
    def __init__(self, conversation_window, no_history):
        self.conversation_window = conversation_window
        self.no_history = no_history

        self.conversation = []
        self.frames = []
        self.reset()
    
    def reset(self):
        # Reset the context when starting a new scenario
        self.conversation = []
    
    def update(self, new_bubble):
        if self.no_history is True:
            return
        # Update context
        self.conversation.append(new_bubble)
        if new_bubble.frame_number not in self.frames:
            self.frames.append(new_bubble.frame_number)
    
    def fifo(self):
        if len(self.frames) >= self.conversation_window:
            self.clean_old_bubbles(self.frames[-self.conversation_window])
    
    def clean_old_bubbles(self, threshold):
        self.conversation[:] = [x for x in self.conversation if x.frame_number > threshold]
    
    def get_context_for_question(self, qid, prev, inherit, frame_number):
        ret_context = [x for x in self.conversation if x.frame_number == frame_number and x.qid in prev.get(qid, []) or \
                                                       x.frame_number < frame_number and x.qid in inherit.get(qid, [])]
        return ret_context
    
    def __str__(self):
        s = "Context(\n"
        for bubble in self.conversation:
            s += f"    [{bubble.scenario}, frame {bubble.frame_number}] {bubble.actor}: {bubble.get_full_words()}\n"
            s += f"    [{bubble.scenario}, frame {bubble.frame_number}] images: {bubble.images}, extra_images: {bubble.extra_images}\n"
        s += ")"
        return s

def create_query(words, images, frame_number, scenario, qid, gt, transform=None, extra_words=None, extra_images=[]):
    new_bubble = Bubble(
        actor='User', words=words, images=images,
        frame_number=frame_number, scenario=scenario,
        extra_words=extra_words, extra_images=extra_images,
        qid=qid, gt=gt, timestamp=datetime.now().timestamp(),
        transform=transform
    )
    return new_bubble

def create_response(words, frame_number, scenario, qid, gt):
    new_bubble = Bubble(
        actor='VLM', words=words, images=[],
        frame_number=frame_number, scenario=scenario,
        extra_words=None, extra_images=[],
        qid=qid, gt=gt, timestamp=datetime.now().timestamp(),
        transform=None
    )
    return new_bubble

class Bubble:
    def __init__(self, actor, words, images, frame_number, scenario, 
                 transform=None, extra_words=None, extra_images=[],
                 qid=-1, gt=None, timestamp=None):
        self.actor = actor
        self.words = words
        self.images = images if images is not None else []
        self.frame_number = frame_number
        self.scenario = scenario
        self.extra_words = extra_words # When asking, some extra information may given by dataset
        self.extra_images = extra_images if extra_images is not None else [] # When asking, some extra information may given by dataset
        self.qid = qid
        self.gt = gt # gt is attached to corresponding Q & A for convenience
        self.timestamp = timestamp
        self.transform = transform

    def get_full_words(self):
        if self.extra_words is not None:
            return self.extra_words + self.words
        else:
            return self.words
        
    def get_full_images(self):
        image_dict = {}

        for images in self.images:
            image_dict[images['frame_number']] = {}
            for key, value in images.items():
                if key != 'frame_number':
                    image_dict[images['frame_number']][key] = value

        for images in self.extra_images:
            image_dict[images['frame_number']] = {}
            for key, value in images.items():
                if key != 'frame_number':
                    image_dict[images['frame_number']][key] = value

        return image_dict
    
    def __str__(self):
        s = "Bubble(\n"
        s += f"    qid = {self.qid}\n"
        s += f"    actor = {self.actor}\n"
        s += f"    full_words = {self.get_full_words()}\n"
        s += f"    full_images = {self.get_full_images()}\n"
        s += f"    frame_number = {self.frame_number}\n"
        s += f"    scenatio = {self.scenario}\n"
        s += f"    gt = {self.gt}\n"
        s += ")"
        return s
    
    def to_dict(self):
        return {
            "actor": self.actor,
            "words": self.words,
            "images": self.images,
            "frame_number": self.frame_number,
            "scenario": self.scenario,
            "extra_words": self.extra_words,
            "extra_images": self.extra_images,
            "qid": self.qid,
            "gt": self.gt,
            "timestamp": self.timestamp,
            "transform": self.transform
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            actor=data["actor"],
            words=data["words"],
            images=data.get("images", []),
            frame_number=data["frame_number"],
            scenario=data["scenario"],
            transform=data.get("transform"),
            extra_words=data.get("extra_words"),
            extra_images=data.get("extra_images", []),
            qid=data.get("qid", -1),
            gt=data.get("gt"),
            timestamp=data.get("timestamp")
        )
