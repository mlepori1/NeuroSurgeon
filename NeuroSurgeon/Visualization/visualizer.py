from collections import defaultdict
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle


class VisualizerConfig:
    """Config object that determines the behavior of the Visualizer
        :param state_dicts: A list of one or two CircuitModel state dicts. Provide 1 to plot subnetwork distribution throughout model. Provide 2 to also include the overlap between subnetworks.
        :type state_dicts: List
        :param model_labels: String labels associated with each CircuitModel state dict, defaults to ["Subnetwork 1", "Subnetwork 2"]
        :type model_labels: List[str], optional
        :param subnetwork_colors: Colors representing each CircuitModel in the diagram, defaults to ["red", "blue"]
        :type subnetwork_colors: List[str], optional
        :param intersect_color: Color representing the intersection of two CircuitModels, defaults to "purple"
        :type intersect_color: str, optional
        :param unused_color: Color representing weights/neurons that are unused in both CircuitModels, defaults to "grey"
        :type unused_color: str, optional
        :param visualize_bias: Whether to also include the bias terms when computing subnetwork distributions throughout the model, defaults to False
        :type visualize_bias: bool, optional
        :param plot_full_network: Whether to only plot layers where at least one model is masked, or whether to plot all layers, defaults to True
        :type plot_full_network: bool, optional
        :param condense_layers: Whether to plot every layer (True), or each parameter tensor seperately (False), defaults to False
        :type condense_layers: bool, optional
        :param mask_method: The mask method used by the CircuitModels, defaults to "continuous_sparsification"
        :type mask_method: str, optional
        :param outfile: The name of the output image file, defaults to "plot.png"
        :type outfile: str, optional
        :param format: The format of the output image file, defaults to "png"
        :type format: str, optional
        :param figsize: The size of the output figure, defaults to (10.0, 20.0)
        :type figsize: Tuple[float], optional
        :param title_fontsize: The fontsize of the title, defaults to 24
        :type title_fontsize: int, optional
        :param label_fontsize: The fontsize of the labels, defaults to 18
        :type label_fontsize: int, optional
        :param alpha: The opacity of the colors, defaults to 0.75
        :type alpha: float, optional
        :param title: The title of the figure, defaults to "Layer By Layer Subnetwork Distribution"
        :type title: str, optional
        :raises ValueError: Raises ValueError if more than 2 state dicts are provided
        """
    def __init__(
        self,
        state_dicts: List,
        model_labels: List[str] = ["Subnetwork 1", "Subnetwork 2"],
        subnetwork_colors: List[str] = ["red", "blue"],
        intersect_color: str = "purple",
        unused_color: str = "grey",
        visualize_bias: bool = False,
        plot_full_network: bool = True,
        condense_layers: bool = False,
        mask_method: str = "continuous_sparsification",
        outfile: str = "plot.png",
        format: str = "png",
        figsize: Tuple[float] = (10.0, 20.0),
        title_fontsize: int = 24,
        label_fontsize: int = 18,
        alpha: float = 0.75,
        title: str = "Layer By Layer Subnetwork Distribution",
    ):
        if len(state_dicts) > 2:
            raise ValueError(
                "Currently only supports plotting one or two subnetworks at a time"
            )
        self.state_dicts = state_dicts
        self.model_labels = model_labels
        self.subnetwork_colors = subnetwork_colors
        self.intersect_color = intersect_color
        self.unused_color = unused_color
        self.visualize_bias = visualize_bias
        self.plot_full_network = plot_full_network
        self.condense_layers = condense_layers
        self.mask_method = mask_method
        self.mask_threshold = (
            0
            if self.mask_method == "continuous_sparsification"
            or self.mask_method == "hard_concrete"
            else None
        )
        self.outfile = outfile
        self.format = format
        self.figsize = figsize
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.alpha = alpha
        self.title = title


class Visualizer:
    """A Class that produces figures describing the distribution of CircuitModel weights
    throughout a model

    :param config: A configuration object describing how to plot the figures
    :type config: VisualizerConfig
    """

    def __init__(self, config):
        self.config = config

    def _filter_state_dicts(self, config):
        if config.condense_layers:
            keys_to_plot = self._filter_layers(config)
        else:
            keys_to_plot = self._filter_nodes(config)
        return keys_to_plot

    def _filter_layers(self, config):
        layers_to_plot = []

        # Filter to just get layers that have at least one masked component at least one model
        if not config.plot_full_network:
            # Get all layers that have masked components in at least one model
            for state_dict in config.state_dicts:
                keys = list(state_dict.keys())
                for key in keys:
                    if "_mask_params" in key:
                        key = key.split(".")
                        layer = -1
                        for component in key:
                            if component.isdigit():
                                layer = component
                        layers_to_plot.append(layer)
        else:
            # Get all layers
            for state_dict in config.state_dicts:
                keys = list(state_dict.keys())
                for key in keys:
                    key = key.split(".")
                    layer = -1
                    for component in key:
                        if component.isdigit():
                            layer = component
                    layers_to_plot.append(layer)

        layers_to_plot = list(set(layers_to_plot))

        # For each model, extract either the masked_params key (if available) or weight key for each key in the relevant layers
        filtered_keys = []
        for state_dict in config.state_dicts:
            model_keys = list(state_dict.keys())
            filtered_model_keys = []
            for key in model_keys:
                if any([layer in key.split(".") for layer in layers_to_plot]):
                    if key + "_mask_params" in model_keys:
                        filtered_model_keys.append(key + "_mask_params")
                    elif "_mask_params" not in key:
                        filtered_model_keys.append(key)

            if self.config.visualize_bias == False:
                filtered_model_keys = [
                    key for key in filtered_model_keys if "bias" not in key
                ]
            filtered_keys.append(filtered_model_keys)

        return filtered_keys

    def _filter_nodes(self, config):
        keys_to_plot = []

        # Filter to just get nodes that are masked in at least one model
        if not config.plot_full_network:
            # Get all nodes that are masked in at least one model
            for state_dict in config.state_dicts:
                keys = list(state_dict.keys())
                keys_to_plot += [key for key in keys if "_mask_params" in key]
            keys_to_plot = [key.replace("_mask_params", "") for key in keys_to_plot]

        else:
            for state_dict in config.state_dicts:
                keys = list(state_dict.keys())
                # Get all keys except for mask params
                keys_to_plot += [key for key in keys if "mask_params" not in key]

        keys_to_plot = list(set(keys_to_plot))

        if self.config.visualize_bias == False:
            keys_to_plot = [key for key in keys_to_plot if "bias" not in key]

        # For each model, extract either the masked_params key (if available) or weight key for each key in all_masked_keys
        filtered_keys = []
        for state_dict in config.state_dicts:
            model_keys = list(state_dict.keys())
            filtered_model_keys = []
            for key in model_keys:
                if key in keys_to_plot:
                    if key + "_mask_params" in model_keys:
                        filtered_model_keys.append(key + "_mask_params")
                    else:
                        filtered_model_keys.append(key)
            filtered_keys.append(filtered_model_keys)

        return filtered_keys

    def _create_layer2nodes(self, plot_nodes, config):
        # Maps layers to the nodes inside of those layers
        model_layer2node_dicts = []
        for model in plot_nodes:
            layer2nodes = defaultdict(list)
            for node in model:
                node_list = node.split(".")
                layer = -1
                for component in node_list:
                    if component.isdigit():
                        layer = component
                # Do not plot embedding layers or poolers
                if layer != -1:
                    layer2nodes[layer].append(node)
            model_layer2node_dicts.append(layer2nodes)
        return model_layer2node_dicts

    def _get_layer2masks(self, model_layer2node_dicts, config):
        # Maps layers to masks for each node in that layer
        model_layer2mask_dicts = []
        for i in range(len(config.state_dicts)):
            state_dict = config.state_dicts[i]
            layer2node = model_layer2node_dicts[i]
            layer2mask = {}
            for layer, nodes in layer2node.items():
                masks = []
                for node in nodes:
                    if "mask_params" not in node:
                        # If model used the full node, create a mask of ones
                        masks.append((node, torch.ones(state_dict[node].shape)))
                    else:
                        # Else create a binary mask from the raw mask parameters
                        masks.append(
                            (node, state_dict[node] > self.config.mask_threshold)
                        )
                layer2mask[layer] = masks
            model_layer2mask_dicts.append(layer2mask)
        return model_layer2mask_dicts

    def _compute_overlap(self, model_layer2node_dicts, config):
        # Compute subnetwork overlap for plotting.
        # If condense_layers: Returns a dictionary mapping layer to overlap data points
        # Else: Returns a dictionary mapping layer to a list of tuples (nodename, overlap data points)
        model_layer2mask_dicts = self._get_layer2masks(model_layer2node_dicts, config)
        if config.condense_layers == False:
            overlap_dict = self._compute_node_overlap(model_layer2mask_dicts, config)
        else:
            overlap_dict = self._compute_layer_overlap(model_layer2mask_dicts, config)
        return overlap_dict

    def _compute_node_overlap(self, model_layer2mask_dicts, config):
        # Computes the overlap of inidividual weight matrices between subnetworks
        overlap_dict = {}
        if len(config.state_dicts) == 2:
            model_0 = model_layer2mask_dicts[0]
            model_1 = model_layer2mask_dicts[1]

            for layer in model_0.keys():
                layer_overlap_data = []
                model_0_nodes = model_0[layer]
                model_1_nodes = model_1[layer]

                for node_idx in range(len(model_0_nodes)):
                    node_name = model_0_nodes[node_idx][0]
                    node_name = ".".join(node_name.split(".")[-3:-1])
                    total = torch.sum(torch.ones(model_0_nodes[node_idx][1].shape))
                    subnetwork_1 = torch.sum(model_0_nodes[node_idx][1])

                    intersection = torch.sum(
                        torch.logical_and(
                            model_0_nodes[node_idx][1], model_1_nodes[node_idx][1]
                        )
                    )
                    subnetwork_2 = torch.sum(model_1_nodes[node_idx][1]) - intersection
                    subnetwork_1 = subnetwork_1 - intersection
                    unused = total - (subnetwork_1 + subnetwork_2 + intersection)
                    node_overlap = {
                        "model_1": subnetwork_1 / total,
                        "model_2": subnetwork_2 / total,
                        "intersection": intersection / total,
                        "unused": unused / total,
                    }
                    layer_overlap_data.append((node_name, node_overlap))
                overlap_dict[layer] = layer_overlap_data
        else:
            model_0 = model_layer2mask_dicts[0]
            for layer in model_0.keys():
                layer_overlap_data = []
                model_0_nodes = model_0[layer]

                for node_idx in range(len(model_0_nodes)):
                    node_name = model_0_nodes[node_idx][0]

                    total = torch.sum(torch.ones(model_0_nodes[node_idx][1].shape))
                    subnetwork_1 = torch.sum(model_0_nodes[node_idx][1])
                    unused = total - (subnetwork_1)
                    node_overlap = {
                        "model_1": subnetwork_1 / total,
                        "model_2": 0 / total,
                        "intersection": 0 / total,
                        "unused": unused / total,
                    }
                    layer_overlap_data.append((node_name, node_overlap))
                overlap_dict[layer] = layer_overlap_data

        return overlap_dict

    def _compute_layer_overlap(self, model_layer2mask_dicts, config):
        # Computes the overlap of full layers between subnetworks
        overlap_dict = {}
        if len(config.state_dicts) == 2:
            model_0 = model_layer2mask_dicts[0]
            model_1 = model_layer2mask_dicts[1]

            for layer in model_0.keys():
                model_0_nodes = model_0[layer]
                model_1_nodes = model_1[layer]

                layer_subnetwork_1 = 0.0
                layer_subnetwork_2 = 0.0
                layer_intersection = 0.0
                layer_unused = 0.0
                layer_total = 0.0

                for node_idx in range(len(model_0_nodes)):
                    total = torch.sum(torch.ones(model_0_nodes[node_idx][1].shape))
                    subnetwork_1 = torch.sum(model_0_nodes[node_idx][1])

                    intersection = torch.sum(
                        torch.logical_and(
                            model_0_nodes[node_idx][1], model_1_nodes[node_idx][1]
                        )
                    )
                    subnetwork_2 = torch.sum(model_1_nodes[node_idx][1]) - intersection
                    subnetwork_1 = subnetwork_1 - intersection
                    unused = total - (subnetwork_1 + subnetwork_2 + intersection)

                    layer_subnetwork_1 += subnetwork_1
                    layer_subnetwork_2 += subnetwork_2
                    layer_intersection += intersection
                    layer_unused += unused
                    layer_total += total

                node_overlap = {
                    "model_1": layer_subnetwork_1 / layer_total,
                    "model_2": layer_subnetwork_2 / layer_total,
                    "intersection": layer_intersection / layer_total,
                    "unused": layer_unused / layer_total,
                }
                overlap_dict[layer] = node_overlap
        else:
            model_0 = model_layer2mask_dicts[0]
            for layer in model_0.keys():
                model_0_nodes = model_0[layer]

                layer_subnetwork_1 = 0.0
                layer_unused = 0.0
                layer_total = 0.0

                for node_idx in range(len(model_0_nodes)):
                    total = torch.sum(torch.ones(model_0_nodes[node_idx][1].shape))
                    subnetwork_1 = torch.sum(model_0_nodes[node_idx][1])
                    unused = total - (subnetwork_1)
                    layer_subnetwork_1 += subnetwork_1
                    layer_unused += unused
                    layer_total += total

                node_overlap = {
                    "model_1": subnetwork_1 / total,
                    "model_2": 0 / total,
                    "intersection": 0 / total,
                    "unused": unused / total,
                }
                overlap_dict[layer] = node_overlap

        return overlap_dict

    def _create_blank_diagram(self, config):
        fig = plt.figure(figsize=(config.figsize))
        ax = fig.add_axes((0, 0, 1, 1))
        # ax.set_xlim(0, config.figsize[0])
        # ax.set_ylim(0, config.figsize[1])
        ax.tick_params(bottom=False, top=False, left=False, right=False)
        ax.tick_params(
            labelbottom=False, labeltop=False, labelleft=False, labelright=False
        )
        return fig, ax

    def _plot_nodes_in_layer(self, ax, overlap, config):
        # Used to plot node-level information

        rectangle_width = 0.75  # Rectangles should be .75 as wide as the image
        node_height = 1 / (
            (1.5 * len(overlap)) + 0.5
        )  # nodes should have equal height, with .5 nodes in between, and on margins
        vertical_space_height = (
            0.5 * node_height
        )  # Should have some space between nodes

        node_heights = []

        for node_idx in range(len(overlap)):
            node = overlap[node_idx]
            node_name = node[0]
            overlaps = node[1]

            current_node_height = (
                1
                - ((node_idx + 1) * node_height)
                - ((node_idx + 1) * vertical_space_height)
            )
            node_heights.append(current_node_height)

            subnetwork_1_width = overlaps["model_1"] * rectangle_width
            subnetwork_2_width = overlaps["model_2"] * rectangle_width
            intersection_width = overlaps["intersection"] * rectangle_width
            unused_width = overlaps["unused"] * rectangle_width

            base_x = 0.5 - (rectangle_width / 2)
            ax.add_patch(
                Rectangle(
                    (base_x, current_node_height),
                    subnetwork_1_width,
                    node_height,
                    color=config.subnetwork_colors[0],
                    alpha=config.alpha,
                )
            )
            ax.add_patch(
                Rectangle(
                    (base_x + subnetwork_1_width, current_node_height),
                    subnetwork_2_width,
                    node_height,
                    color=config.subnetwork_colors[1],
                    alpha=config.alpha,
                )
            )
            ax.add_patch(
                Rectangle(
                    (
                        base_x + subnetwork_1_width + subnetwork_2_width,
                        current_node_height,
                    ),
                    intersection_width,
                    node_height,
                    color=config.intersect_color,
                    alpha=config.alpha,
                )
            )
            ax.add_patch(
                Rectangle(
                    (
                        base_x
                        + subnetwork_1_width
                        + subnetwork_2_width
                        + intersection_width,
                        current_node_height,
                    ),
                    unused_width,
                    node_height,
                    color=config.unused_color,
                    alpha=config.alpha,
                )
            )

            ax.text(
                0.5,
                current_node_height + 0.5 * node_height,
                node_name,
                horizontalalignment="center",
                verticalalignment="center",
                color="black",
                fontsize=config.label_fontsize,
            )

        for idx in range(len(node_heights)):
            if idx != 0:
                ax.annotate(
                    "",
                    (0.5, node_heights[idx] + node_height),
                    (0.5, node_heights[idx - 1]),
                    arrowprops=dict(arrowstyle="->"),
                )

    def _create_graph(self, node_overlaps, config):
        # Inspired by the tutorial: https://matplotlib.org/matplotblog/posts/mpl-for-making-diagrams/
        fig, ax = self._create_blank_diagram(config)

        rectangle_width = 0.75  # Rectangles should be .75 as wide as the image
        layer_height = 1 / (
            (1.5 * len(node_overlaps.keys())) + 0.5
        )  # Layers should have equal height, with .5 layer_height in between, and on margins
        vertical_space_height = (
            0.5 * layer_height
        )  # Should have some space between layers

        # Graph layers in order
        layers = list(node_overlaps.keys())
        layers = [int(layer) for layer in layers]
        layers.sort()
        layers = [str(layer) for layer in layers]

        ax.text(
            0.5,
            1 - (0.5 * vertical_space_height),
            config.title,
            horizontalalignment="center",
            verticalalignment="center",
            color="black",
            fontsize=config.title_fontsize,
        )

        layer_heights = []
        for layer_idx in range(len(layers)):
            if config.condense_layers:
                layer = layers[layer_idx]
                current_layer_height = (
                    1
                    - ((layer_idx + 1) * layer_height)
                    - ((layer_idx + 1) * vertical_space_height)
                )
                layer_heights.append(current_layer_height)
                overlaps = node_overlaps[layer]
                subnetwork_1_width = overlaps["model_1"] * rectangle_width
                subnetwork_2_width = overlaps["model_2"] * rectangle_width
                intersection_width = overlaps["intersection"] * rectangle_width
                unused_width = overlaps["unused"] * rectangle_width

                base_x = 0.5 - (rectangle_width / 2)
                ax.add_patch(
                    Rectangle(
                        (base_x, current_layer_height),
                        subnetwork_1_width,
                        layer_height,
                        color=config.subnetwork_colors[0],
                        alpha=config.alpha,
                    )
                )
                ax.add_patch(
                    Rectangle(
                        (base_x + subnetwork_1_width, current_layer_height),
                        subnetwork_2_width,
                        layer_height,
                        color=config.subnetwork_colors[1],
                        alpha=config.alpha,
                    )
                )
                ax.add_patch(
                    Rectangle(
                        (
                            base_x + subnetwork_1_width + subnetwork_2_width,
                            current_layer_height,
                        ),
                        intersection_width,
                        layer_height,
                        color=config.intersect_color,
                        alpha=config.alpha,
                    )
                )
                ax.add_patch(
                    Rectangle(
                        (
                            base_x
                            + subnetwork_1_width
                            + subnetwork_2_width
                            + intersection_width,
                            current_layer_height,
                        ),
                        unused_width,
                        layer_height,
                        color=config.unused_color,
                        alpha=config.alpha,
                    )
                )

                ax.text(
                    0.1,
                    current_layer_height + 0.5 * layer_height,
                    f"Layer: {layer}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    rotation=90,
                    color="black",
                    fontsize=config.label_fontsize,
                )
            else:
                layer = layers[layer_idx]
                current_layer_height = (
                    1
                    - ((layer_idx + 1) * layer_height)
                    - ((layer_idx + 1) * vertical_space_height)
                )
                layer_heights.append(current_layer_height)

                # Define a new set of axes just within this layer block
                base_x = 0.5 - (rectangle_width / 2)
                layer_ax = fig.add_axes(
                    (base_x, current_layer_height, rectangle_width, layer_height)
                )
                layer_ax.tick_params(bottom=False, top=False, left=False, right=False)
                layer_ax.tick_params(
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False
                )
                layer_ax.set_facecolor("lightgrey")

                overlaps = node_overlaps[layer]
                self._plot_nodes_in_layer(layer_ax, overlaps, config)

                ax.text(
                    0.1,
                    current_layer_height + 0.5 * layer_height,
                    f"Layer: {layer}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    rotation=90,
                    color="black",
                    fontsize=config.label_fontsize,
                )

        for idx in range(len(layer_heights)):
            if idx != 0:
                ax.annotate(
                    "",
                    (0.5, layer_heights[idx] + layer_height),
                    (0.5, layer_heights[idx - 1]),
                    arrowprops=dict(arrowstyle="->"),
                )
        plt.savefig(config.outfile, format=config.format)

    def plot(self):
        """The main function of this class. Plots the CircuitModels according to
        the VisualizerConfig specification.
        """
        # First filter state dicts to figure out what to plot
        plot_nodes = self._filter_state_dicts(self.config)
        # Next, organized filtered nodes into layer : node_name dictionary
        layer2nodes = self._create_layer2nodes(plot_nodes, self.config)
        # Then compute the overlap at the desired level of granularity
        node_overlaps = self._compute_overlap(layer2nodes, self.config)
        # Then plot nodes
        self._create_graph(node_overlaps, self.config)
