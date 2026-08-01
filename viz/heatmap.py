import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

from Models.base import CausalModel


def _cmi(model: CausalModel):
    cmi = model.get_causal_graph().cpu().numpy().copy()
    for i in range(cmi.shape[0]):
        cmi[i, i] = 0.0
    return cmi


def _draw_heatmap(model: CausalModel, cmi, threshold, with_slider=False):
    fd = cmi.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    if with_slider:
        plt.subplots_adjust(bottom=0.3)

    image = axes[0].imshow(cmi, cmap="hot", aspect="auto")
    axes[0].set_xticks(range(cmi.shape[1]))
    axes[0].set_yticks(range(fd))
    axes[0].set_xticklabels(model.node_names + ["action"], rotation=45, ha="right", fontsize=9)
    axes[0].set_yticklabels(model.node_names, fontsize=9)
    axes[0].set_title("CMI Heatmap (raw values)")
    axes[0].set_xlabel("Source")
    axes[0].set_ylabel("Target")
    plt.colorbar(image, ax=axes[0])

    binary = (cmi >= threshold).astype(float)
    binary_image = axes[1].imshow(binary, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    axes[1].set_xticks(range(cmi.shape[1]))
    axes[1].set_yticks(range(fd))
    axes[1].set_xticklabels(model.node_names + ["action"], rotation=45, ha="right", fontsize=9)
    axes[1].set_yticklabels(model.node_names, fontsize=9)
    title = axes[1].set_title(f"Binary Graph (threshold={threshold})")
    axes[1].set_xlabel("Source")
    axes[1].set_ylabel("Target")
    plt.colorbar(binary_image, ax=axes[1])
    return fig, binary_image, title


def visualize_cmi_heatmap(model: CausalModel, threshold=None):
    threshold = model.cmi_threshold if threshold is None else threshold
    _draw_heatmap(model, _cmi(model), threshold)
    plt.show()


def select_cmi_threshold(model: CausalModel):
    cmi = _cmi(model)
    fig, binary_image, title = _draw_heatmap(model, cmi, model.cmi_threshold, with_slider=True)

    ax_slider = plt.axes([0.25, 0.12, 0.5, 0.03])
    slider = widgets.Slider(
        ax_slider,
        "Threshold",
        valmin=0.0,
        valmax=float(cmi.max()),
        valinit=model.cmi_threshold,
        valstep=0.01,
    )
    ax_button = plt.axes([0.45, 0.04, 0.1, 0.04])
    button = widgets.Button(ax_button, "Confirm")

    def update(_):
        threshold = slider.val
        binary = (cmi >= threshold).astype(float)
        binary_image.set_data(binary)
        title.set_text(f"Binary Graph (threshold={threshold:.2f})")
        fig.canvas.draw_idle()

    selected_threshold = model.cmi_threshold

    def confirm(_):
        nonlocal selected_threshold
        selected_threshold = slider.val
        plt.close(fig)

    slider.on_changed(update)
    button.on_clicked(confirm)
    plt.show()
    print(f"Selected CMI Threshold: {selected_threshold:.2f}")
    return selected_threshold
