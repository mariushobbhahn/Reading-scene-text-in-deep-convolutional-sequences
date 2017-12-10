
export enum Name {
  jan = 0,
  roman = 1,
  marius = 2
}

export interface Profile {
  name: string;
}

export const profiles: Profile[] = [
  {
    name: 'Jan Haug'
  },
  {
    name: 'Roman Schulte'
  },
  {
    name: 'Marius Hobbhahn'
  }
];
