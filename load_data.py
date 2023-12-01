import os

def dataloader(pathtofile,annotfilename):
    entries = dict()

    #data: 1:ambulance  2:Bus  3:Car 4:Motorcycle  5:Truck
    #coco: 3:car  4:motorcycle  6:bus 8:truck
    #mapper: 1,2->6  3->3  4->4  5->8
    mapper = {'Ambulance': 6, 'Bus': 6, 'Car': 3, 'Motorcycle': 4, 'Truck': 8}

    csv_path=os.path.join(pathtofile,annotfilename)
    with open(csv_path, 'r') as csvfile:
        lines = csvfile.read().splitlines()

        for line in lines[1:]:  # Skip the header
            rowdata = line.strip().split(',')
            filename, class_label =rowdata[0], rowdata[3]
            new_label = mapper[class_label]
            bbox_coordinates = list(map(int, rowdata[4:]))

            if filename not in entries:
                entries[filename] = { 'class' : [], 'bbox_coordinates' : [] }

            entries[filename]['class'].append(new_label)
            entries[filename]['bbox_coordinates'].append(bbox_coordinates)

    return entries