import torch

class CollateFN:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """
        list_idx = []
        list_coordinates = []
        list_features = []
        list_labels = []
        for d in list_data:
            list_coordinates.append(d["coordinates"].to(self.device))
            list_features.append(d["features"].to(self.device))
            list_labels.append(d["labels"])
            list_idx.append(d["idx"].view(-1, 1))

        coordinates_batch = torch.cat(list_coordinates, dim=0)
        features_batch = torch.cat(list_features, dim=0)
        labels_batch = torch.cat(list_labels, dim=0)
        idx = torch.cat(list_idx, dim=0)
        return {"coordinates": coordinates_batch,
                "features": features_batch,
                "labels": labels_batch,
                "idx": idx}

class CollateMerged:
    def __init__(self,
                 device: torch.device = torch.device("cpu")) -> None:
        self.device = device

    def __call__(self, list_data) -> dict:
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """

        source_list_data = [(d["source_coordinates"].to(self.device), d["source_features"].to(self.device), d["source_labels"]) for d in list_data]
        target_list_data = [(d["target_coordinates"].to(self.device), d["target_features"].to(self.device), d["target_labels"]) for d in list_data]

        source_coordinates_batch, source_features_batch, source_labels_batch = convertListToBatch(source_list_data)

        target_coordinates_batch, target_features_batch, target_labels_batch = convertListToBatch(target_list_data)

        return_dict = {"source_coordinates": source_coordinates_batch,
                       "source_features": source_features_batch,
                       "source_labels": source_labels_batch,
                       "target_coordinates": target_coordinates_batch,
                       "target_features": target_features_batch,
                       "target_labels": target_labels_batch}

        return return_dict


def convertListToBatch(list_data):
    list_coordinates = []
    list_features = []
    list_labels = []
    for d in list_data:
        list_coordinates.append(d[0])
        list_features.append(d[1])
        list_labels.append(d[2])
    coordinates_batch = torch.cat(list_coordinates, dim=0)
    features_batch = torch.cat(list_features, dim=0)
    labels_batch = torch.cat(list_labels, dim=0)
    return coordinates_batch, features_batch, labels_batch