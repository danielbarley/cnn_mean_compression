import pandas as pd
import optparse

parser = optparse.OptionParser()
parser.add_option("--file", type=str, default="./res/mem1.csv")
parser.add_option("--blocksize", type=int, default=1)
args, options = parser.parse_args()

df = pd.read_csv(
    args.file,
    names=[
        "name",
        "bs",
        "after_init",
        "after_model",
        "after_input",
        "after_forward",
        "after_loss",
        "after_backward",
        "after_optim",
        "rel_model",
        "rel_input",
        "rel_forward",
        "rel_loss",
        "rel_backward",
        "rel_optim",
        "rel_peak",
    ],
)

for field in [
    "after_init",
    "after_model",
    "after_input",
    "after_forward",
    "after_loss",
    "after_backward",
    "after_optim",
]:
    df[field] = df[field] / 1024**3

print([x for x in zip(df.T.values[0], df.T.values[9])])
print([x for x in zip(df.T.values[0], df.T.values[10])])
print([x for x in zip(df.T.values[0], df.T.values[11])])
archs = [
    "resnet",
    # "vgg",
    # "densenet",
    # "swin",
    # "vit",
    # "mobilenet",
    # "efficientnet",
]
for arch in archs:
    df.where(df["name"].str.contains(arch, regex=False))[
        [
            "name",
            "after_model",
            "after_input",
            "after_forward",
            "after_backward",
            "after_optim",
        ]
    ].dropna().T.reset_index(drop=True).to_csv(
        f"{arch}_{args.blocksize}.dat",
        index=True,
        index_label="index",
        sep="\t",
        header=False,
    )

df["perc_model"] = df["rel_model"] / df["rel_peak"] * 100
df["perc_input"] = df["rel_input"] / df["rel_peak"] * 100
df["perc_forward"] = df["rel_forward"] / df["rel_peak"] * 100
df[["name", "perc_model", "perc_input", "perc_forward"]].to_markdown(
    f"percentages_{args.blocksize}.md"
)
# print(df["perc_forward"].mean())
# print(df["perc_forward"].min())
# print(df["perc_forward"].max())
