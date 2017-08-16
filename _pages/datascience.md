---
layout: archive
permalink: /datascience/
title: "Data Science Posts by Tags"
author_profile: true
header:
  image: "renwick_string.jpg"
  caption: "Photo credit: Ginny Lehman"

---

{% include base_path %}
{% include group-by-array collection=site.posts field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}