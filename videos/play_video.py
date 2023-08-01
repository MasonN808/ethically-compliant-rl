from base64 import b64encode
from IPython.display import HTML
def render_mp4(videopath: str) -> str:
  """
  Gets a string containing a b4-encoded version of the MP4 video
  at the specified path.
  """
  mp4 = open(videopath, 'rb').read()
  base64_encoded_mp4 = b64encode(mp4).decode()
  return f'<video width=400 controls><source src="data:video/mp4;' \
         f'base64,{base64_encoded_mp4}" type="video/mp4"></video>'
if __name__=="__main__":
    html = render_mp4("/Users/masonnakamura/Local-Git/ethically-compliant-rl/videos/ppo_train.mp4")
    HTML(html)