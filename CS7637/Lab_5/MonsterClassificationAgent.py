class MonsterClassificationAgent:
    def __init__(self):
        """
        size: tiny, small, medium, large, huge
        color: black, white, brown, gray, red, yellow, blue, green, orange, purple
        covering: fur, feathers, scales, skin
        foot-type: paw, hoof, talon, foot, none
        leg-count: 0, 1, 2, 3, 4, 5, 6, 7, 8
        arm-count: 0, 1, 2, 3, 4, 5, 6, 7, 8
        eye-count: 0, 1, 2, 3, 4, 5, 6, 7, 8
        horn-count: 0, 1, 2
        lays-eggs: true, false
        has-wings: true, false
        has-gills: true, false
        has-tail: true, false
        """
        self.general_model = {
            "size": set(["tiny", "small", "medium", "large", "huge"]),
            "color": set(["black", "white", "brown", "gray", "red", "yellow", "blue", "green", "orange", "purple"]),
            "covering": set(["fur", "feathers", "scales", "skin"]),
            "foot-type": set(["paw", "hoof", "talon", "foot", "none"]),
            "leg-count": set(range(0, 9)),
            "arm-count": set(range(0, 9)),
            "eye-count": set(range(0, 9)),
            "horn-count": set(range(0, 3)),
            "lays-eggs": set([True, False]),
            "has-wings": set([True, False]),
            "has-gills": set([True, False]),
            "has-tail": set([True, False])
        }
        self.specific_model = {}



    def solve(self, samples, new_monster):
        #Add your code here!
        #
        #The first parameter to this method will be a labeled list of samples in the form of
        #a list of 2-tuples. The first item in each 2-tuple will be a dictionary representing
        #the parameters of a particular monster. The second item in each 2-tuple will be a
        #boolean indicating whether this is an example of this species or not.
        #
        #The second parameter will be a dictionary representing a newly observed monster.
        #
        #Your function should return True or False as a guess as to whether or not this new
        #monster is an instance of the same species as that represented by the list.

        for monster, is_positive in samples:
            if is_positive:
                for key, value in monster.items():
                    if key not in self.specific_model:
                        self.specific_model[key] = set()
                    self.specific_model[key].add(value)

        # Look for essential features in specific model that define the species
        # Keys that have less than 2 unique values in the positive samples are considered essential and appear more than once in the positive samples.
        

        for key, value in new_monster.items():
            if key in self.specific_model and value not in self.specific_model[key]:
                return False

        return True