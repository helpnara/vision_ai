import { imageUrl, pathUrl } from "../api";

export interface GalleryItem {
  /** manifest image_id (imageUrl로 서빙) 또는 */
  imageId?: string;
  /** 프로젝트 폴더 안의 파일 경로 (pathUrl로 서빙) */
  path?: string;
  /** 이미 완성된 URL */
  src?: string;
  caption?: string;
  width?: number;
}

/** 4열 이미지 격자 (st.columns(4) + st.image 의 자리). */
export function Gallery({ items, cols = 4 }: { items: GalleryItem[]; cols?: 2 | 3 | 4 | 5 }) {
  if (items.length === 0) return null;
  return (
    <div className={`gallery cols-${cols}`}>
      {items.map((item, i) => {
        const src = item.src ?? (item.imageId ? imageUrl(item.imageId, item.width ?? 480) : item.path ? pathUrl(item.path, item.width ?? 480) : "");
        return (
          <figure key={`${src}-${i}`}>
            <img src={src} alt={item.caption ?? ""} loading="lazy" />
            {item.caption ? <figcaption>{item.caption}</figcaption> : null}
          </figure>
        );
      })}
    </div>
  );
}
