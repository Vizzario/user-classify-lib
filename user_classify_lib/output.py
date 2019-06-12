"""
Customized library to format the results from CorEx and generate graphs based on the results.

Author: Eric Hu

"""
import numpy as np

class LatexChartGenerator():

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "w")

    # region Header
    def writeHeader(self):
        self.f.write(
            """% \\BEGIN_FOLD and END_FOLD used by TeXstudio but have no effect in Overleaf

\\documentclass[class=minimal, border=8pt]{standalone}

% Packages
\\usepackage[utf8]{inputenc}
\\usepackage[dvipsnames]{xcolor}

\\usepackage{tikz}
\\usetikzlibrary{positioning, backgrounds, fit, calc}

%BEGIN_FOLD Set up Constants
\\newcommand{\\innerSep}{0.15cm} % Text buffer around text in nodes and around nodes in surround
\\newcommand{\\columnXShift}{5cm + 3cm + 2 * \\innerSep} % Column title width plus separation plus buffer
\\newcommand{\\dashOpacity}{0.0} % Set greater than zero to see dashed guidelines
%END_FOLD

%BEGIN_FOLD Set up Colors
\\definecolor{Layer1ItemOutline}{rgb}{0.25, 0.75, 0.25}
\\definecolor{Layer1ItemFill}{rgb}{0.75, 1.0, 0.75}
\\definecolor{Layer1ItemOutlineSuppressed}{rgb}{0.5, 1.0, 0.5}
\\definecolor{Layer1ItemFillSuppressed}{rgb}{0.9, 1.0, 0.9}
\\definecolor{Layer1GroupOutline}{rgb}{0.0, 0.25, 0.0}
\\definecolor{Layer1GroupFill}{rgb}{0.95, 1.0, 0.95}

\\definecolor{Layer2ItemOutline}{rgb}{0.25, 0.25, 0.75}
\\definecolor{Layer2ItemFill}{rgb}{0.75, 0.75, 1.0}
\\definecolor{Layer2ItemOutlineSuppressed}{rgb}{0.5, 0.5, 1.0}
\\definecolor{Layer2ItemFillSuppressed}{rgb}{0.9, 0.9, 1.0}
\\definecolor{Layer2GroupOutline}{rgb}{0.0, 0.0, 0.25}
\\definecolor{Layer2GroupFill}{rgb}{0.95, 0.95, 1.0}
%END_FOLD

%BEGIN_FOLD Set up TikZ styles
% Titles
\\tikzstyle{documentTitle} = [
	align=center,
	font=\\fontsize{16}{0}\\bfseries\\selectfont
]
\\tikzstyle{columnTitle} = [
    text width=5cm,
    align=left,
    font=\\bfseries, % Bold
    inner sep=\\innerSep,
    rounded corners,
    dashed,
    draw=black,
    draw opacity=\\dashOpacity
]
\\tikzstyle{groupTitle} = [
    text width=8cm,
    align=left,
    font=\\itshape, % Italics
    inner sep=\\innerSep,
    rounded corners,
    dashed,
    draw=black,
    draw opacity=\\dashOpacity
]
% Visible Items
\\tikzstyle{Layer1Item} = [
    align=center,
    shape=rectangle,
    rounded corners,
    inner sep=\\innerSep,
    draw=Layer1ItemOutline,
    fill=Layer1ItemFill
]
\\tikzstyle{Layer2Item} = [
    align=center,
    shape=rectangle,
    rounded corners,
    inner sep=\\innerSep,
    draw=Layer2ItemOutline,
    fill=Layer2ItemFill
]
% Suppressed Items
\\tikzstyle{Layer1ItemSuppressed} = [
    align=center,
    shape=rectangle,
    rounded corners,
    inner sep=\\innerSep,
    draw=Layer1ItemOutlineSuppressed,
    fill=Layer1ItemFillSuppressed,
    text=textSuppressed
]
\\tikzstyle{Layer2ItemSuppressed} = [
    align=center,
    shape=rectangle,
    rounded corners,
    inner sep=\\innerSep,
    draw=Layer2ItemOutlineSuppressed,
    fill=Layer2ItemFillSuppressed,
    text=textSuppressed
]
% Surrounds
\\tikzstyle{Layer1GroupSurround} = [
    shape=rectangle,
    rounded corners,
    inner sep=\\innerSep,
    draw=Layer1GroupOutline,
    fill=Layer1GroupFill
]
\\tikzstyle{Layer2GroupSurround} = [
    shape=rectangle,
    rounded corners,
    inner sep=\\innerSep,
    draw=Layer2GroupOutline,
    fill=Layer2GroupFill
]
% Arrows
\\tikzstyle{arrow} = [
	thick,
	->,
	>=stealth,
	rounded corners = 5pt
]
\\tikzstyle{arrowSuppressed} = [
	thick,
	->,
	>=stealth,
	opacity=0.1,
	rounded corners = 5pt
]
\\tikzstyle{arrowHidden} = [
    opacity=0
]
%END_FOLD
% Begin document
\\begin{document}

% Begin TikZ flow-chart environment
\\begin{tikzpicture}[
    node distance=1cm % Used as default vertical spacing between adjacent nodes
]

% Title
\\node (documentTitleNode) [documentTitle, anchor=center] at (2.5, 1) {Clustering Results from CorEx Analysis};"""
        )

    # endregion

    # region Footer
    def writeFooter(self):
        self.f.write(
            """
            % End TikZ flow-chart environment
\\end{tikzpicture}

% End document
\\end{document}
            """
        )
        self.f.close()
    # endregion

    #region Clusters

    def writeClusters(self, clusters: np.ndarray, n_clusters: int, layer: int = 0, cluster_order=None, tc : float = None):
        tc_value = "N/A"
        if tc is not None:
            tc_value = str(tc)

        if layer == 1:
            self.f.write("""\n%BEGIN_FOLD Column Nodes: Layer {0}
\\node (Layer{0}ColumnTitle) [columnTitle] at (0, 0) {{Layer {0}, Total TC: {1}}};""".format(layer, tc_value))
        else:
            self.f.write("""\n%BEGIN_FOLD Column Nodes: Layer {0}
\\node (Layer{0}ColumnTitle) [columnTitle, right=\\columnXShift*80 of Layer{1}ColumnTitle.center, anchor=center] {{Layer {0}, Total TC: {2}}};""".format(layer, str(layer-1), tc_value))

        previous_cluster = "Layer{0}ColumnTitle".format(layer)

        for j in range(n_clusters):

            if cluster_order is None:
                c_num = j
            else:
                c_num = cluster_order[j]
            cluster_name = "Layer{0}GroupCluster{1}".format(layer, c_num)
            cluster_title = "Cluster {0}".format(c_num)
            self.f.write("\n% {0}".format(cluster_title))
            if j == 0:
                self.f.write("\n\\node ({0}) [groupTitle,below=of {1}.south west, anchor=west] {{{2}}};".format(cluster_name, previous_cluster, cluster_title))
            else:
                self.f.write(
                    "\n\\node ({0}) [groupTitle,below=of {1}.south west, anchor=west, xshift=\\innerSep] {{{2}}};".format(cluster_name,
                                                                                                       previous_cluster,
                                                                                                       cluster_title))
            self.f.write("\n\\foreach \\name/\\type/\\below/\\text in {")
            previous_node = cluster_name
            cluster_header = "Layer{0}".format(layer)
            for k in range(len(clusters[c_num])):
                if k == 0:
                    self.f.write("\n\t{0}{1}/Layer{2}Item/{3}/{4}".format(cluster_header, str(clusters[c_num][k]).replace(" ", "_").replace("|", "_"), layer, previous_node, str(clusters[c_num][k]).replace("_", " ")))
                else:
                    self.f.write(",\n\t{0}{1}/Layer{2}Item/{3}/{4}".format(cluster_header, str(clusters[c_num][k]).replace(" ", "_").replace("|", "_"), layer,
                                                                          previous_node,
                                                                          str(clusters[c_num][k]).replace("_", " ")))
                previous_node = "{0}{1}".format(cluster_header, str(clusters[c_num][k].replace(" ", "_").replace("|", "_")))
            self.f.write("""}}
{{
	\\node (\\name) [\\type, below of=\\below] {{\\text}};
}}
\\begin{{scope}}[on background layer]
	\\node ({0}Surround) [Layer{1}GroupSurround, fit = ({0}) ({2})] {{}};
\end{{scope}}
            
            """.format(cluster_name, layer, previous_node))
            previous_cluster = cluster_name + "Surround"

        self.f.write("%END_FOLD")



    #endregion

    #region Arrows
    def writeArrows(self, source_n_clusters, source_layer_name, source_layer=1, dest_layer = 2):
        self.f.write("\n%BEGIN_FOLD Arrows")
        self.f.write("\n\\foreach \\type/\start/\\finish[count=\\count from 0] in {")
        for i in range(source_n_clusters):
            if i == 0:
                self.f.write("\n\tarrow/Layer{0}GroupCluster{1}Surround/Layer{2}{3}".format(source_layer, i, dest_layer, source_layer_name + str(i)))
            else:
                self.f.write(",\n\tarrow/Layer{0}GroupCluster{1}Surround/Layer{2}{3}".format(source_layer, i, dest_layer,
                                                                                            source_layer_name + str(
                                                                                                i)))
        self.f.write("""}
{
	\\draw [\\type]
		(\start.east) --
		(\\finish.west);
}
%END_FOLD""")

    #endregion