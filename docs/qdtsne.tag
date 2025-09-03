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
    <templarg>std::size_t num_dim_</templarg>
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
      <type>Index_</type>
      <name>num_observations</name>
      <anchorfile>classqdtsne_1_1Status.html</anchorfile>
      <anchor>a77523d2a0d5aaad8db433efc2dd8c4cc</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>run</name>
      <anchorfile>classqdtsne_1_1Status.html</anchorfile>
      <anchor>a04b541369258a61abd5bef8d89dc5d7f</anchor>
      <arglist>(Float_ *const Y, const int limit)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>run</name>
      <anchorfile>classqdtsne_1_1Status.html</anchorfile>
      <anchor>aee7b3da21eb4af05d26e5acff71cbd7b</anchor>
      <arglist>(Float_ *const Y)</arglist>
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
      <anchor>ac2bbf464d20eaeefd7df7ddc881affaa</anchor>
      <arglist>(const std::size_t data_dim, const Index_ num_obs, const Float_ *const data, const knncolle::Builder&lt; Index_, Float_, Float_, Matrix_ &gt; &amp;builder, const Options &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>Index_</type>
      <name>perplexity_to_k</name>
      <anchorfile>namespaceqdtsne.html</anchorfile>
      <anchor>a74ac9034c403a269170cf638d46b95e5</anchor>
      <arglist>(const double perplexity)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>initialize_random</name>
      <anchorfile>namespaceqdtsne.html</anchorfile>
      <anchor>aef04d8843d9a4a611868107efc6990a7</anchor>
      <arglist>(Float_ *const Y, const std::size_t num_points, const unsigned long long seed=42)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; Float_ &gt;</type>
      <name>initialize_random</name>
      <anchorfile>namespaceqdtsne.html</anchorfile>
      <anchor>a320f37a0f1ba77edd186158a967f8a9b</anchor>
      <arglist>(const std::size_t num_points, const unsigned long long seed=42)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>parallelize</name>
      <anchorfile>namespaceqdtsne.html</anchorfile>
      <anchor>a1d3f2500838138042a25dc8081a30e10</anchor>
      <arglist>(const int num_workers, const Task_ num_tasks, Run_ run_task_range)</arglist>
    </member>
  </compound>
  <compound kind="page">
    <name>index</name>
    <title>Quick and dirty t-SNE in C++</title>
    <filename>index.html</filename>
    <docanchor file="index.html">md__2github_2workspace_2README</docanchor>
  </compound>
</tagfile>
