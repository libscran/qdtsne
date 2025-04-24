<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<tagfile doxygen_version="1.12.0">
  <compound kind="file">
    <name>initialize.hpp</name>
    <path>qdtsne/</path>
    <filename>initialize_8hpp.html</filename>
    <includes id="Status_8hpp" name="Status.hpp" local="yes" import="no" module="no" objc="no">Status.hpp</includes>
    <includes id="Options_8hpp" name="Options.hpp" local="yes" import="no" module="no" objc="no">Options.hpp</includes>
    <namespace>qdtsne</namespace>
  </compound>
  <compound kind="file">
    <name>Options.hpp</name>
    <path>qdtsne/</path>
    <filename>Options_8hpp.html</filename>
    <class kind="struct">qdtsne::Options</class>
    <namespace>qdtsne</namespace>
  </compound>
  <compound kind="file">
    <name>qdtsne.hpp</name>
    <path>qdtsne/</path>
    <filename>qdtsne_8hpp.html</filename>
    <includes id="Options_8hpp" name="Options.hpp" local="yes" import="no" module="no" objc="no">Options.hpp</includes>
    <includes id="initialize_8hpp" name="initialize.hpp" local="yes" import="no" module="no" objc="no">initialize.hpp</includes>
    <includes id="Status_8hpp" name="Status.hpp" local="yes" import="no" module="no" objc="no">Status.hpp</includes>
    <includes id="utils_8hpp" name="utils.hpp" local="yes" import="no" module="no" objc="no">utils.hpp</includes>
    <namespace>qdtsne</namespace>
  </compound>
  <compound kind="file">
    <name>Status.hpp</name>
    <path>qdtsne/</path>
    <filename>Status_8hpp.html</filename>
    <includes id="Options_8hpp" name="Options.hpp" local="yes" import="no" module="no" objc="no">Options.hpp</includes>
    <includes id="utils_8hpp" name="utils.hpp" local="yes" import="no" module="no" objc="no">utils.hpp</includes>
    <class kind="class">qdtsne::Status</class>
    <namespace>qdtsne</namespace>
  </compound>
  <compound kind="file">
    <name>utils.hpp</name>
    <path>qdtsne/</path>
    <filename>utils_8hpp.html</filename>
    <namespace>qdtsne</namespace>
  </compound>
  <compound kind="struct">
    <name>qdtsne::Options</name>
    <filename>structqdtsne_1_1Options.html</filename>
    <member kind="variable">
      <type>double</type>
      <name>perplexity</name>
      <anchorfile>structqdtsne_1_1Options.html</anchorfile>
      <anchor>ae485ba64f2ff5379f09a75fb89447628</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>infer_perplexity</name>
      <anchorfile>structqdtsne_1_1Options.html</anchorfile>
      <anchor>a824d9599d7ea23dcfdffc871afca0d49</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>theta</name>
      <anchorfile>structqdtsne_1_1Options.html</anchorfile>
      <anchor>ac69fb8679d8fe13ee31e5e06d7425cc6</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>max_iterations</name>
      <anchorfile>structqdtsne_1_1Options.html</anchorfile>
      <anchor>a5caa0c9124703fd6220dfb51e0aa10e1</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>stop_lying_iter</name>
      <anchorfile>structqdtsne_1_1Options.html</anchorfile>
      <anchor>af352392082ebac2046b5679b218d6594</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>mom_switch_iter</name>
      <anchorfile>structqdtsne_1_1Options.html</anchorfile>
      <anchor>a0efc1892882f473e5e38e5ab11d5d604</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>start_momentum</name>
      <anchorfile>structqdtsne_1_1Options.html</anchorfile>
      <anchor>ab5e4ad0d2dc1daf15e48d193bb3c3d8f</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>final_momentum</name>
      <anchorfile>structqdtsne_1_1Options.html</anchorfile>
      <anchor>aade824422084ecfb009f600c38c58abe</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>eta</name>
      <anchorfile>structqdtsne_1_1Options.html</anchorfile>
      <anchor>a7f0dcada1addd28c257f0074727f9eae</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>exaggeration_factor</name>
      <anchorfile>structqdtsne_1_1Options.html</anchorfile>
      <anchor>ab0f96457fa0b6d05dd32b8ab3fdf31b6</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>max_depth</name>
      <anchorfile>structqdtsne_1_1Options.html</anchorfile>
      <anchor>a3202c4085c933353f0fbfb9cc1825f64</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>leaf_approximation</name>
      <anchorfile>structqdtsne_1_1Options.html</anchorfile>
      <anchor>a69b189230017a5574b41d2cac7780656</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>num_threads</name>
      <anchorfile>structqdtsne_1_1Options.html</anchorfile>
      <anchor>abf708aa5e462b8b3038fcaef2711cbb0</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>qdtsne::Status</name>
    <filename>classqdtsne_1_1Status.html</filename>
    <templarg>int num_dim_</templarg>
    <templarg>typename Index_</templarg>
    <templarg>typename Float_</templarg>
    <member kind="function">
      <type>int</type>
      <name>iteration</name>
      <anchorfile>classqdtsne_1_1Status.html</anchorfile>
      <anchor>a2e5fbb69b89a43b2deea5c46374b642b</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>max_iterations</name>
      <anchorfile>classqdtsne_1_1Status.html</anchorfile>
      <anchor>af67e41bd74305fbeba6f486bda76813f</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>num_observations</name>
      <anchorfile>classqdtsne_1_1Status.html</anchorfile>
      <anchor>ae4090be4f7e29149a6bf9c420d8e8b14</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>run</name>
      <anchorfile>classqdtsne_1_1Status.html</anchorfile>
      <anchor>a56714ad4a50f6a467599fbcf2ce69d51</anchor>
      <arglist>(Float_ *Y, int limit)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>run</name>
      <anchorfile>classqdtsne_1_1Status.html</anchorfile>
      <anchor>a73f99b3c064d4054396373543cb57820</anchor>
      <arglist>(Float_ *Y)</arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>qdtsne</name>
    <filename>namespaceqdtsne.html</filename>
    <class kind="struct">qdtsne::Options</class>
    <class kind="class">qdtsne::Status</class>
    <member kind="typedef">
      <type>knncolle::NeighborList&lt; Index_, Float_ &gt;</type>
      <name>NeighborList</name>
      <anchorfile>namespaceqdtsne.html</anchorfile>
      <anchor>a06453be45faea3c21c0a66a0564bd084</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>Status&lt; num_dim_, Index_, Float_ &gt;</type>
      <name>initialize</name>
      <anchorfile>namespaceqdtsne.html</anchorfile>
      <anchor>ac7386895ff09abfb4c792a0ffdd9e170</anchor>
      <arglist>(NeighborList&lt; Index_, Float_ &gt; neighbors, const Options &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>Status&lt; num_dim_, Index_, Float_ &gt;</type>
      <name>initialize</name>
      <anchorfile>namespaceqdtsne.html</anchorfile>
      <anchor>ae63cd82791c5b13f110050b28d3fffbd</anchor>
      <arglist>(const knncolle::Prebuilt&lt; Index_, Input_, Float_ &gt; &amp;prebuilt, const Options &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>Status&lt; num_dim_, Index_, Float_ &gt;</type>
      <name>initialize</name>
      <anchorfile>namespaceqdtsne.html</anchorfile>
      <anchor>af2c49636abb5292d2d7e3756e7dc2a78</anchor>
      <arglist>(std::size_t data_dim, Index_ num_points, const Float_ *data, const knncolle::Builder&lt; Index_, Float_, Float_, Matrix_ &gt; &amp;builder, const Options &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>perplexity_to_k</name>
      <anchorfile>namespaceqdtsne.html</anchorfile>
      <anchor>a673dd037719d1b1824da29e20fd1618d</anchor>
      <arglist>(double perplexity)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>initialize_random</name>
      <anchorfile>namespaceqdtsne.html</anchorfile>
      <anchor>a916236f6f1d1f805b1e68f2a11ae4296</anchor>
      <arglist>(Float_ *Y, size_t num_points, int seed=42)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; Float_ &gt;</type>
      <name>initialize_random</name>
      <anchorfile>namespaceqdtsne.html</anchorfile>
      <anchor>aca9df1b3ededa7ddc62d3cfa0225341c</anchor>
      <arglist>(size_t num_points, int seed=42)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>parallelize</name>
      <anchorfile>namespaceqdtsne.html</anchorfile>
      <anchor>abfd3b53975731fd9df83eac1cc99fada</anchor>
      <arglist>(int num_workers, Task_ num_tasks, Run_ run_task_range)</arglist>
    </member>
  </compound>
  <compound kind="page">
    <name>index</name>
    <title>C++ library for t-SNE</title>
    <filename>index.html</filename>
    <docanchor file="index.html">md__2github_2workspace_2README</docanchor>
  </compound>
</tagfile>
