---
permalink: /
title: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

<div style="text-align: justify; font-size: 90%;"> 
I am a graduated MSc student from <a href="https://www.ucl.ac.uk">University College London</a> focusing on rendering, computer vision and machine learning.
</div>

<br/>
<p style="font-size: 35px; font-weight: 700;">Publications</p>
<ul>
  {% for post in site.publications reversed %}
    {% unless post.hidden == true %}
      {% include archive-single.html %}
    {% endunless %}
  {% endfor %}
</ul>